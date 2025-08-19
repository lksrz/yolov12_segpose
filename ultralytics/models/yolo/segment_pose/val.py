# AGPL-3.0 License

from multiprocessing.pool import ThreadPool
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import LOGGER, NUM_THREADS, ops
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.metrics import SegmentPoseMetrics, OKS_SIGMA
from ultralytics.utils.plotting import output_to_target, plot_images


class SegmentPoseValidator(DetectionValidator):
    """Validator for the combined segmentation + pose task."""

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.args.task = "segment_pose"
        self.metrics = SegmentPoseMetrics(save_dir=self.save_dir, on_plot=self.on_plot)
        self.plot_masks = []

    def preprocess(self, batch):
        batch = super().preprocess(batch)
        batch["masks"] = batch["masks"].to(self.device).float()
        batch["keypoints"] = batch["keypoints"].to(self.device).float()
        return batch

    def init_metrics(self, model):
        super().init_metrics(model)
        self.kpt_shape = self.data["kpt_shape"]
        # Prepare sigma for reuse of PoseValidator _process_batch
        is_pose = self.kpt_shape == [17, 3]
        nkpt = self.kpt_shape[0]
        self.sigma = OKS_SIGMA if is_pose else np.ones(nkpt) / nkpt
        self.stats = dict(tp_m=[], tp_p=[], tp=[], conf=[], pred_cls=[], target_cls=[], target_img=[])

    def get_desc(self):
        # 1 label + 3*4 metrics columns = 15 fields; '%22s' + 14 metrics
        return ("%22s" + "%11s" * 14) % (
            "Class",
            "Images",
            "Instances",
            "Box(P",
            "R",
            "mAP50",
            "mAP50-95)",
            "Mask(P",
            "R",
            "mAP50",
            "mAP50-95)",
            "Pose(P",
            "R",
            "mAP50",
            "mAP50-95)",
        )

    def postprocess(self, preds):
        """
        Extract predictions and proto tensor from SegmentPose model inference output.
        
        SegmentPose model outputs:
        - Non-export PyTorch: (concat_predictions, (det[1], mc, p, kpt_raw))
        - Exported format: similar to segment model format
        """
        if isinstance(preds, tuple) and len(preds) == 2:
            predictions, aux = preds
            if isinstance(aux, (tuple, list)):
                if len(aux) >= 3:
                    # PyTorch format: aux = (det[1], mc, p, kpt_raw) - proto at index 2
                    proto = aux[2]
                elif len(aux) == 1:
                    # Exported format: aux is just the proto tensor
                    proto = aux[0]
                else:
                    proto = None
            else:
                # aux is directly the proto tensor
                proto = aux
        else:
            # Fallback - single tensor
            predictions = preds[0] if isinstance(preds, (list, tuple)) else preds
            proto = None
            
        p = ops.non_max_suppression(
            predictions,
            self.args.conf,
            self.args.iou,
            labels=self.lb,
            multi_label=True,
            agnostic=self.args.single_cls or self.args.agnostic_nms,
            max_det=self.args.max_det,
            nc=self.nc,
        )
        return p, proto

    def _prepare_batch(self, si, batch):
        pbatch = super()._prepare_batch(si, batch)
        midx = [si] if self.args.overlap_mask else batch["batch_idx"] == si
        pbatch["masks"] = batch["masks"][midx]
        kpts = batch["keypoints"][batch["batch_idx"] == si]
        h, w = pbatch["imgsz"]
        kpts = kpts.clone()
        kpts[..., 0] *= w
        kpts[..., 1] *= h
        kpts = ops.scale_coords(pbatch["imgsz"], kpts, pbatch["ori_shape"], ratio_pad=pbatch["ratio_pad"])
        pbatch["kpts"] = kpts
        return pbatch

    def _prepare_pred(self, pred, pbatch, proto):
        predn = super()._prepare_pred(pred, pbatch)
        # masks & kpts: split tail into [mask_coeffs | keypoints]
        nk, nd = (self.kpt_shape if isinstance(self.kpt_shape, (list, tuple)) else (pbatch["kpts"].shape[1], 3))
        kpt_dims = nk * nd
        
        # Use YOLOv12-seg approach: slice mask coeffs as everything between boxes and keypoints
        # Structure: [box(4) + conf(1) + cls(1) + mask_coeffs(nm) + keypoints(kpt_dims)]
        mask_start = 6
        mask_end = pred.shape[1] - kpt_dims
        mc = pred[:, mask_start:mask_end]
        nm = mc.shape[1]  # Actual number of mask coefficients
        
        # Validate against proto if available
        proto_for_masks = proto
        if proto is not None and nm > 0:
            if nm != proto.shape[0]:
                # Truncate to proto size for safety if mismatch
                nm_actual = min(nm, proto.shape[0])
                mc = mc[:, :nm_actual]
            if proto.dim() != 3:
                # Proto tensor should be 3D (nm, h, w) - skip mask processing if invalid
                proto_for_masks = None
        # If proto is None, mask evaluation will be skipped
        
        # Use pre-scaled boxes for mask projection in model space
        pred_masks = (
            ops.process_mask_native(proto_for_masks, mc, pred[:, :4], shape=pbatch["imgsz"]) if proto_for_masks is not None else None
        )
        # Keypoints from scaled preds
        pred_kpts = predn[:, -kpt_dims:].view(len(predn), nk, nd) if kpt_dims > 0 else torch.zeros((len(predn), nk, nd), device=predn.device, dtype=predn.dtype)
        ops.scale_coords(pbatch["imgsz"], pred_kpts, pbatch["ori_shape"], ratio_pad=pbatch["ratio_pad"])
        return predn, pred_masks, pred_kpts

    def update_metrics(self, preds, batch):
        # preds is (predictions_list, proto_tensor) from postprocess
        predictions, proto_tensor = preds[0], preds[1]  
        for si, pred in enumerate(predictions):
            # proto_tensor is shared across batch, index by batch item if 4D, use as-is if 3D
            if proto_tensor is not None:
                if proto_tensor.dim() == 4:  # (B, nm, h, w)
                    if si < proto_tensor.shape[0]:
                        proto = proto_tensor[si]
                    else:
                        # Use last proto if batch index exceeds proto batch size
                        proto = proto_tensor[-1]
                else:  # (nm, h, w) - single proto for all batch items
                    proto = proto_tensor
            else:
                # Skip mask processing if proto is None
                proto = None
            self.seen += 1
            npr = len(pred)
            stat = dict(
                conf=torch.zeros(0, device=self.device),
                pred_cls=torch.zeros(0, device=self.device),
                tp=torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device),
                tp_m=torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device),
                tp_p=torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device),
            )
            pbatch = self._prepare_batch(si, batch)
            cls, bbox = pbatch.pop("cls"), pbatch.pop("bbox")
            nl = len(cls)
            stat["target_cls"] = cls
            stat["target_img"] = cls.unique()
            if npr == 0:
                if nl:
                    for k in self.stats.keys():
                        self.stats[k].append(stat[k])
                    if self.args.plots:
                        self.confusion_matrix.process_batch(detections=None, gt_bboxes=bbox, gt_cls=cls)
                continue

            if self.args.single_cls:
                pred[:, 5] = 0
            predn, pred_masks, pred_kpts = self._prepare_pred(pred, pbatch, proto)
            stat["conf"] = predn[:, 4]
            stat["pred_cls"] = predn[:, 5]

            if nl:
                stat["tp"] = self._process_batch(predn, bbox, cls)
                if pred_masks is not None:
                    # Reuse segmentation validator logic for mask IoUs
                    from ultralytics.models.yolo.segment.val import SegmentationValidator
                    stat["tp_m"] = SegmentationValidator._process_batch(
                        self, predn, bbox, cls, pred_masks, pbatch["masks"], self.args.overlap_mask, masks=True
                    )
                else:
                    # No predicted masks - all zeros
                    stat["tp_m"] = torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device)
                # Reuse pose validator logic for keypoint OKS
                from ultralytics.models.yolo.pose.val import PoseValidator
                stat["tp_p"] = PoseValidator._process_batch(self, predn, bbox, cls, pred_kpts, pbatch["kpts"]) 
            if self.args.plots:
                self.confusion_matrix.process_batch(predn, bbox, cls)

            for k in self.stats.keys():
                self.stats[k].append(stat[k])

            if self.args.plots and self.batch_i < 3 and pred_masks is not None:
                self.plot_masks.append(torch.as_tensor(pred_masks, dtype=torch.uint8)[:15].cpu())

    def finalize_metrics(self, *args, **kwargs):
        # Set speed into combined metrics object
        self.metrics.speed = self.speed

    def get_stats(self):
        stats = {k: torch.cat(v, 0).cpu().numpy() for k, v in self.stats.items()}
        self.nt_per_class = np.bincount(stats["target_cls"].astype(int), minlength=self.nc)
        self.nt_per_image = np.bincount(stats["target_img"].astype(int), minlength=self.nc)
        stats.pop("target_img", None)
        if len(stats) and (stats.get("tp") is not None) and (stats["tp"].any()):
            # Reorder to match SegmentPoseMetrics.process signature
            self.metrics.process(stats["tp"], stats.get("tp_m", []), stats.get("tp_p", []), stats["conf"], stats["pred_cls"], stats["target_cls"])
        return self.metrics.results_dict


    def plot_predictions(self, batch, preds, ni):
        """Plot predicted detections only (boxes/conf/class) for segment_pose to avoid tuple handling issues."""
        plot_images(
            batch["img"],
            *output_to_target(preds[0], max_det=self.args.max_det),
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_pred.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )

