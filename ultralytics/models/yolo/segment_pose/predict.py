# AGPL-3.0 License

from ultralytics.engine.results import Results
from ultralytics.models.yolo.detect import DetectionPredictor
from ultralytics.utils import ops


class SegmentPosePredictor(DetectionPredictor):
    """Predictor for combined segmentation + pose task."""

    def postprocess(self, preds, img, orig_imgs):
        """Return list[Results] with boxes, masks and keypoints for segment_pose task."""
        if not isinstance(preds, (list, tuple)):
            return super().postprocess(preds, img, orig_imgs)

        # y: (B, 6+nm+kpt_dims, N) output; aux: (feats, mc, proto) from head
        y, aux = preds
        proto = aux[2] if isinstance(aux, (list, tuple)) and len(aux) >= 3 else None

        p = ops.non_max_suppression(
            y,
            self.args.conf,
            self.args.iou,
            labels=getattr(self, "lb", []),
            multi_label=True,
            agnostic=self.args.single_cls or self.args.agnostic_nms,
            max_det=self.args.max_det,
            nc=len(self.model.names),
        )

        results = []
        for i, pred in enumerate(p):
            orig_img = orig_imgs[i] if isinstance(orig_imgs, list) else orig_imgs
            path = self.batch[0]
            img_path = path[i] if isinstance(path, list) else path

            if len(pred) == 0:
                # Return empty boxes (0,6) to satisfy Results' Boxes shape expectations
                empty_boxes = pred[:, :6]
                results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=empty_boxes))
                continue

            # Use YOLOv12-seg approach: slice mask coeffs as everything between boxes and keypoints
            nk, nd = self.model.kpt_shape
            kpt_dims = nk * nd
            # Structure: [box(4) + conf(1) + cls(1) + mask_coeffs(nm) + keypoints(kpt_dims)]
            mask_start = 6
            mask_end = pred.shape[1] - kpt_dims
            mc = pred[:, mask_start:mask_end]
            nm = mc.shape[1]  # Actual number of mask coefficients
            
            kpt_tail = pred[:, -kpt_dims:] if kpt_dims > 0 else pred.new_zeros((len(pred), 0))
            
            # Validate against proto if available
            if proto is not None and nm > 0 and nm != proto.shape[1]:
                print(f"Warning: Calculated nm={nm} != proto channels={proto.shape[1]} - truncating to match")
                nm_actual = min(nm, proto.shape[1])
                mc = mc[:, :nm_actual]

            # masks
            if self.args.retina_masks:
                masks = ops.process_mask_native(proto[i], mc, pred[:, :4], orig_img.shape[:2]) if proto is not None else None
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            else:
                masks = ops.process_mask(proto[i], mc, pred[:, :4], img.shape[2:], upsample=True) if proto is not None else None
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)

            # keypoints
            pred_kpts = kpt_tail.view(len(pred), nk, nd)
            pred_kpts = ops.scale_coords(img.shape[2:], pred_kpts, orig_img.shape)

            results.append(
                Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6], masks=masks, keypoints=pred_kpts)
            )

        return results


