from ultralytics import YOLO

model = YOLO('src/model/yolov8m.pt')

# Train the model
model.train(data='xView.yaml', epochs=100, imgsz=640)