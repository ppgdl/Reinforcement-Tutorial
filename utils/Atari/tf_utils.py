import cv2
import os

def rgb2gray(image):
    gray = cv2.cvtColor(image, cv2.COLOR_EGB2HSV)

    return gray


def resizeimage(image, height, width):
    image = cv2.resize(image, (height, width))
    gray = rgb2gray(image)

    return gray


def save_image(image, config, index):
    if not os.path.exists(config.imgpath):
        os.makedirs(config.imgpath)

    path = os.path.join(config.imgpath, str(index) + '.png')

    cv2.imwrite(path, image)