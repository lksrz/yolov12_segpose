# AGPL-3.0 License

from copy import copy

from ultralytics.models import yolo
from ultralytics.nn.tasks import SegmentPoseModel
from ultralytics.utils import DEFAULT_CFG, RANK
from ultralytics.utils.plotting import plot_images, plot_results


class SegmentPoseTrainer(yolo.detect.DetectionTrainer):
    """Trainer for combined segmentation + pose task."""

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        if overrides is None:
            overrides = {}
        overrides["task"] = "segment_pose"
        super().__init__(cfg, overrides, _callbacks)

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = SegmentPoseModel(cfg, ch=3, nc=self.data["nc"], data_kpt_shape=self.data["kpt_shape"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        from .val import SegmentPoseValidator

        self.loss_names = ("box_loss", "seg_loss", "pose_loss", "kobj_loss", "cls_loss", "dfl_loss")
        return SegmentPoseValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks)

    def plot_training_samples(self, batch, ni):
        images = batch["img"]
        cls = batch["cls"].squeeze(-1)
        bboxes = batch["bboxes"]
        paths = batch["im_file"]
        batch_idx = batch["batch_idx"]
        plot_images(
            images,
            batch_idx,
            cls,
            bboxes,
            masks=batch.get("masks"),
            kpts=batch.get("keypoints"),
            paths=paths,
            fname=self.save_dir / f"train_batch{ni}.jpg",
            on_plot=self.on_plot,
        )

    def plot_metrics(self):
        plot_results(file=self.csv, segment=True, pose=True, on_plot=self.on_plot)


