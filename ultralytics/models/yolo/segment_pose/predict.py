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

            # split mask coeffs and kpts tail from pred vector using model's dynamic kpt_shape
            nk, nd = self.model.kpt_shape
            kpt_dims = nk * nd
            nm = int(proto.shape[1]) if proto is not None else max(pred.shape[1] - 6 - kpt_dims, 0)
            assert pred.shape[1] >= 6 + nm + kpt_dims, f"Pred width {pred.shape[1]} < required {6 + nm + kpt_dims}"
            tail_w = max(pred.shape[1] - 6 - nm, 0)
            kpt_tail = pred[:, -kpt_dims:] if tail_w >= kpt_dims else pred.new_zeros((len(pred), kpt_dims))
            mc = pred[:, 6 : 6 + nm]
            # Ensure mask coeffs match proto channels
            if proto is not None and nm > 0:
                assert mc.shape[1] == proto.shape[1], f"Mask coeffs {mc.shape[1]} != proto channels {proto.shape[1]}"

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


