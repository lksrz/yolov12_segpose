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
        p = ops.non_max_suppression(
            preds[0] if isinstance(preds, (list, tuple)) else preds,
            self.args.conf,
            self.args.iou,
            labels=self.lb,
            multi_label=True,
            agnostic=self.args.single_cls or self.args.agnostic_nms,
            max_det=self.args.max_det,
            nc=self.nc,
        )
        # Aux for segment_pose is (feats, mask_coeffs, proto, kpt_raw)
        proto = preds[1][2] if isinstance(preds, (list, tuple)) else None
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
        
        # Use proto channels directly if available, otherwise calculate from pred width
        if proto is not None:
            nm = int(proto.shape[0])
        else:
            # Fallback: calculate from prediction width
            nm = max(0, pred.shape[1] - 6 - kpt_dims)
        
        # Validate the prediction vector can accommodate this nm
        required_width = 6 + nm + kpt_dims
        if pred.shape[1] < required_width:
            print(f"Warning: pred_width={pred.shape[1]} < required={required_width} (nm={nm}, kpt_dims={kpt_dims})")
            # Adjust nm to fit actual prediction
            nm = max(0, pred.shape[1] - 6 - kpt_dims)
        tail_w = max(pred.shape[1] - 6 - nm, 0)
        mc = pred[:, 6 : 6 + nm]
        # Handle mismatch between mask coeffs and proto channels
        proto_for_masks = proto
        if proto is not None and nm > 0 and mc.shape[1] != proto.shape[0]:
            print(f"Warning: Mask coeffs {mc.shape[1]} != proto channels {proto.shape[0]}, truncating to smaller")
            # Use the minimum to avoid indexing errors
            nm_actual = min(mc.shape[1], proto.shape[0])
            mc = mc[:, :nm_actual]  # truncate mask coeffs if needed
            # Handle different proto tensor dimensions
            if proto.dim() == 3:  # (nm, h, w)
                proto_for_masks = proto[:nm_actual, :, :]
            elif proto.dim() == 2:  # malformed old model
                print(f"Warning: Proto has only {proto.dim()}D, expected 3D. Old model compatibility mode.")
                proto_for_masks = None  # disable mask processing for malformed models
            else:
                proto_for_masks = None
        # Use pre-scaled boxes for mask projection in model space
        pred_masks = (
            ops.process_mask_native(proto_for_masks, mc, pred[:, :4], shape=pbatch["imgsz"]) if proto_for_masks is not None else None
        )
        # Keypoints from scaled preds (fallback to zeros if not present in NMS output)
        if tail_w >= kpt_dims:
            pred_kpts = predn[:, -kpt_dims:].view(len(predn), nk, nd)
        else:
            pred_kpts = torch.zeros((len(predn), nk, nd), device=predn.device, dtype=predn.dtype)
        ops.scale_coords(pbatch["imgsz"], pred_kpts, pbatch["ori_shape"], ratio_pad=pbatch["ratio_pad"])
        return predn, pred_masks, pred_kpts

    def update_metrics(self, preds, batch):
        # Extract proto tensor correctly: (feats, mask_coeffs, proto, kpt_raw)
        proto_batch = preds[1][2] if isinstance(preds, (list, tuple)) and len(preds[1]) >= 3 else None
        for si, pred in enumerate(preds[0]):
            proto = proto_batch[si] if proto_batch is not None else None
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

