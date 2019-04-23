from PIL import Image
import os
import tensorflow as tf


def crop_TB_image():
    csv_path = '../BME590_ML_Final_Project/crop_image.py'
    csv_path = os.path.join(os.path.dirname(__file__), csv_path)

    xmin = 830
    ymin = 1029
    xmax = 886
    ymax = 1093
    xc = round((xmax + xmin)/2)
    yc = round((ymax + ymin)/2)
    coords = (xc - 50, xc + 50, yc - 50, yc + 50)
    print(coords)

    image_path = './Original_dataset/tuberculosis-phone-0001.jpg'
    saved_location = './TB_Image/tuberculosis-phone-0001_1.jpg'

    normal_img = crop(image_path, coords, saved_location)

    # random crop
    for i in range(10):
        img_name = ('./Non_TB_Image/{}.jpeg'.format(i))
        img_arr = tf.image.random_crop(normal_img, [100, 100, 3])
        with tf.Session() as session:
            non_tb_img = session.run(img_arr)
        non_tb_img = Image.fromarray(non_tb_img)
        save = check_TB_region(non_tb_img)
        if save == bool(1):
            print('Cropped Image saved')
            non_tb_img.save(img_name)


def crop(image_path, coords, saved_location):
    """
    @param image_path: The path to the image to edit
    @param coords: A tuple of x/y coordinates (x1, y1, x2, y2)
    @param saved_location: Path to save the cropped image
    """
    image_obj = Image.open(image_path)
    cropped_image = image_obj.crop(coords)
    cropped_image.save(saved_location)

    pixelMap = image_obj.load()

    img = image_obj
    pixelsNew = img.load()

    for i in range(img.size[0]):
        if coords[0] <= i <= coords[1]:
            for j in range(img.size[1]):
                if coords[2] <= j <= coords[3]:
                    pixelMap[i, j] = (0, 0, 0, 255)
                else:
                    pixelsNew[i, j] = pixelMap[i, j]
    return img


def check_TB_region(img):
    pixelMap = img.load()
    save = bool(1)
    for i in range(img.size[0]):
            for j in range(img.size[1]):
                if pixelMap[i, j] == (0, 0, 0):
                    save = bool(0)
                    print('Image contains TB region')
    return save


if __name__ == '__main__':
    crop_TB_image()
