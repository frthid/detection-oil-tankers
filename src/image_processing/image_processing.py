from osgeo import gdal
import numpy as np
from object_detection.object_detection import detect_objects

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
            process_tile(tile)


def process_tile(tile):
    transposed_image = np.transpose(tile, (1, 2, 0))
    if not transposed_image.flags['C_CONTIGUOUS']:
        transposed_image = np.ascontiguousarray(transposed_image)
    print("Обработка изображения")
    detect_objects(transposed_image)


# def main():
#     image_path = "src/test_img/NHT.tif"
#     tile_size = 512
#     overlap = 30
#     split_image(image_path, tile_size, overlap)

# main()