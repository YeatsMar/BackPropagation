from PIL import Image, ImageFilter
import numpy as np
import os
import cv2
from matplotlib import pyplot as plt

def image2array(filepath, reverse=False):
    im = Image.open(filepath)
    pixels = im.load()
    for x in range(im.width):
        for y in range(im.height):
            if reverse:
                pixels[x, y] = 0 if pixels[x, y] == 255 else 1
            else:
                pixels[x, y] = 1 if pixels[x, y] == 255 else 0
    pixels = np.array(im.getdata())
    pixels.shape = im.width * im.height
    return pixels


def get_data(reverse=False):
    X = list()
    Y = list()
    for i in range(14):
        for file in os.listdir('TRAIN/%d'%(i+1)):
            if '.bmp' not in file:
                continue
            x = image2array('TRAIN/%d/'%(i+1) + file, reverse)
            X.append(x)
            y = np.zeros(14)
            y[i] = 1
            Y.append(y)
    return (np.array(X), np.array(Y))


def crop_image(filepath, box):
    im = Image.open(filepath)
    im.crop(0, 0, 25, 25)


def blur_gaussian():
    # im = Image.open('TRAIN/1/1.bmp')
    # pixels = np.array(im.getdata())
    # pixels.shape = im.width * im.height
    # print(pixels)
    #
    # im.show()
    # im = im.filter(ImageFilter.SMOOTH) #
    # pixels = np.array(im.getdata())
    # pixels.shape = im.width * im.height
    # print(pixels)
    # im.show()

    im = Image.open('TRAIN/1/1.bmp')
    pixels = im.load()
    for x in range(im.width):
        for y in range(im.height):
            pixels[x, y] = 0 if pixels[x, y] == 255 else 1
    im.show()

    im.filter(ImageFilter.SMOOTH)
    im.show()

if __name__ == '__main__':
    blur_gaussian()
    # img = cv2.imread('TRAIN/1/0.bmp')
    # blur = cv2.GaussianBlur(img, (3, 3), 0)
    #
    # plt.subplot(121), plt.imshow(img), plt.title('Original')
    # plt.xticks([]), plt.yticks([])
    # plt.subplot(122), plt.imshow(blur), plt.title('Blurred')
    # plt.xticks([]), plt.yticks([])
    # plt.show()

