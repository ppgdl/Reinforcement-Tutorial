import numpy as np
import cv2


def rgb2gray(image):
    gray = cv2.cvtColor(image, cv2.COLOR_EGB2GRAY)

    return gray


def resizeimage(image, height, width):

    return cv2.resize(image, (height, width))


