import geojson

# создать список объектов для записи в geojson
features = []
pixel_size = dataset.GetGeoTransform()[1]
geotransform = dataset.GetGeoTransform()

# цикл по блокам изображения из первой программы
for i in range(num_blocks_y):
    for j in range(num_blocks_x):
        # получить координаты верхнего левого угла блока в отдельном tile
        y_start = i * (tile_size - overlap)
        x_start = j * (tile_size - overlap)

        # пересчитать координаты входного файла
        input_y_start = geotransform[3] + y_start * pixel_size
        input_x_start = geotransform[0] + x_start * pixel_size
        input_y_end = geotransform[3] + (y_start + tile_size) * pixel_size + overlap * pixel_size
        input_x_end = geotransform[0] + (x_start + tile_size) * pixel_size + overlap * pixel_size

        # обработать блок изображения и детектировать объекты
        tile = dataset.ReadAsArray(x_start, y_start, tile_size, tile_size)
        results = model.predict(tile, save=True, save_txt=True, conf=0.5)

        # пересчитать координаты рамок (bounding boxes) объектов в координаты входного файла
        for box in results[0].boxes:
            class_id = results[0].names[box.cls[0].item()]
            cords = box.xyxy[0].tolist()
            cords = [round(input_x_start + x * pixel_size) for x in cords]
            conf = round(box.conf[0].item(), 2)

            # создать geojson объект для каждого обнаруженного объекта
            feature = geojson.Feature(
                geometry=geojson.Polygon([[
                    (cords[0], cords[1]),
                    (cords[2], cords[1]),
                    (cords[2], cords[3]),
                    (cords[0], cords[3]),
                    (cords[0], cords[1])
                ]]),
                properties={
                    "class": class_id,
                    "confidence": conf
                }
            )

            # добавить geojson объект в список
            features.append(feature)

# создать geojson FeatureCollection из списка объектов
feature_collection = geojson.FeatureCollection(features)

# записать geojson FeatureCollection в файл
with open("output.geojson", "w") as f:
    geojson.dump(feature_collection, f)