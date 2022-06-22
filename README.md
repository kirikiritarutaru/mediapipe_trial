# mediapipe_trial
[MediaPipe](https://google.github.io/mediapipe/)お試しリポジトリ
- MediaPipe Pose
- MediaPipe Hands

##* 準備
- Pythonライブラリをインストール
  - opencv-python
  - numpy
  - mediapipe
  - numba

- `models`下に`coco.names`を配置
  - [coco.names](https://github.com/AlexeyAB/darknet/blob/master/data/coco.names)

- `models`下に各モデルのcfgとweightsを配置
  - YOLOv4
    - [yolov4.weights](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights)
    - [yolov4.cfg](https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-csp.cfg)
  - YOLOv4-tiny
    - [yolov4-tiny.weights](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights)
    - [yolov4-tiny.cfg](https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg)
