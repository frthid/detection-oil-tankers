from ultralytics import YOLO

model = YOLO('src/model/yolov8m.pt')

model.train(data='src/train_modal/config.yaml', epochs=5, imgsz=640)