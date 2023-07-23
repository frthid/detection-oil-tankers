from osgeo import gdal
import numpy as np
from object_detection.object_detection import detect_objects
from image_processing.coordinates_processed import recording_processed_coordinates

def split_image(image_path, tile_size, overlap):
    dataset = gdal.Open(image_path)
    width = dataset.RasterXSize
    height = dataset.RasterYSize
    num_blocks_x = int(np.ceil((width - overlap) / (tile_size - overlap)))
    num_blocks_y = int(np.ceil((height - overlap) / (tile_size - overlap)))

    for i in range(num_blocks_y):
        for j in range(num_blocks_x):
            y_start = i * (tile_size - overlap)
            x_start = j * (tile_size - overlap)
            y_end = y_start + tile_size
            x_end = x_start + tile_size
            if y_end > height:
                y_end = height
                y_start = y_end - tile_size
            if x_end > width:
                x_end = width
                x_start = x_end - tile_size
            tile = dataset.ReadAsArray(x_start, y_start, tile_size, tile_size)

            transposed_image = np.transpose(tile, (1, 2, 0))
            if not transposed_image.flags['C_CONTIGUOUS']:
                transposed_image = np.ascontiguousarray(transposed_image)

            process_tile(num_blocks_x, num_blocks_y, tile_size, overlap, transposed_image, dataset)

def process_tile(num_blocks_x, num_blocks_y, tile_size, overlap, tile, dataset):
    print("Обработка изображения")
    results = detect_objects(tile)
    recording_processed_coordinates(num_blocks_x, num_blocks_y, tile_size, overlap, tile, results, dataset)

# def main():
#     image_path = "src/test_img/NHT.tif"
#     tile_size = 512
#     overlap = 30
#     split_image(image_path, tile_size, overlap)

# main()