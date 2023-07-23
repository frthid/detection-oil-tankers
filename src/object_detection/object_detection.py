from ultralytics import YOLO
import numpy as np
from pathlib import Path

def detect_objects(source):
  model = YOLO('src/trained_model/best.pt')

  results = model.predict(source, conf=0.5)
  #print(results)
  #print(type(results[0].boxes))
  #print(results[0].boxes)

  # СОХРАНЕНИЕ КАЖДОГО ОБНАРУЖЕННОГО ОБЪЕКТА
  # for i, result in enumerate(results):
  #   if len(result) > 0:
  #     save_dir = Path('src/test_img')
  #     file_name = save_dir / f'result_{i}.jpg'  # Здесь создается путь к файлу с уникальным именем
  #     result.save_crop(save_dir, file_name)
    
  result = results[0]
  print(len(result.boxes))

  # box = result.boxes[0]   
  # for box in result.boxes:
  #   class_id = result.names[box.cls[0].item()]
  #   cords = box.xyxy[0].tolist()
  #   cords = [round(x) for x in cords]
  #   conf = round(box.conf[0].item(), 2)
  #   print("Object type:", class_id)
  #   print("Coordinates:", cords)
  #   print("Probability:", conf)
  #   print("---")

  return results

# #source = np.random.randint(low=0, high=255, size=(640, 1280, 3), dtype='uint8')
#source = 'src/test_img/NHT1.tif'
# #source = 'src/test_img/images(1).jpg'
#print(source)
#detect_objects(source)