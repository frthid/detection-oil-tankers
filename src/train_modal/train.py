from ultralytics import YOLO
from IPython import display

model = YOLO('src/model/yolov8m.pt')
model.train(data='src/train_modal/data.yaml', epochs=30, imgsz=512, plots=True, device=0)

Image(filename='runs/detect/train/confusion_matrix.png', width=800)
Image(filename='runs/detect/train/results.png', width=800)
Image(filename='runs/detect/train/val_batch0_pred.jpg', width=800)