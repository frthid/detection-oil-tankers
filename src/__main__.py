from image_processing.image_processing import split_image
#from object_detection.object_detection import detect_objects

#source = 'src/test_img/01_4_4_jpg.rf.4d7e83079ffab3608dbd1d20eb681d95.jpg'
#detect_objects(source)

def main():
    image_path = "src/test_img/NHT1.tif"
    tile_size = 512
    overlap = 30
    split_image(image_path, tile_size, overlap)

if __name__ == "__main__":
    main()