# Segment+Pose Task (YOLOv12)

Train:

`yolo train model=ultralytics/cfg/models/v12/yolov12-segpose.yaml data=ultralytics/cfg/datasets/coco-segment_pose.yaml`

Val:

`yolo val model=runs/segment_pose/exp/weights/best.pt data=...`

Predict:

`yolo predict model=runs/segment_pose/exp/weights/best.pt source=ultralytics/assets/bus.jpg`

Label format (one line per instance):

`cls cx cy w h kp1x kp1y kp1v ... kpKx kpKy kpKv xseg1 yseg1 xseg2 yseg2 ...`

Use `kpt_shape` in dataset YAML to define K and visibility format.


