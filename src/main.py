from ultralytics import YOLO

model = YOLO('src/model/yolov8m.pt')

source = 'src/data/images/dusa_anton_2.jpg'

results = model.predict(source, save=True, save_txt=True, conf=0.5)
result = results[0]
print(len(result.boxes))

box = result.boxes[0]   
for box in result.boxes:
  class_id = result.names[box.cls[0].item()]
  cords = box.xyxy[0].tolist()
  cords = [round(x) for x in cords]
  conf = round(box.conf[0].item(), 2)
  print("Object type:", class_id)
  print("Coordinates:", cords)
  print("Probability:", conf)
  print("---")

