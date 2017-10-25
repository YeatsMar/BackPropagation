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
            if reverse:
                im = Image.open('TRAIN/%d/'%(i+1) + file)
                X.append(crop_image_part(im, (0, 0, 26, 26)))
                X.append(crop_image_part(im, (2, 0, 28, 26)))
                X.append(crop_image_part(im, (0, 2, 26, 28)))
                X.append(crop_image_part(im, (2, 2, 28, 28)))
                for j in range(4):
                    y = np.zeros(14)
                    y[i] = 1
                    Y.append(y)
    return (np.array(X), np.array(Y))


def crop_image_part(im, box):
    partial = im.crop(box)
    partial = partial.resize((28, 28), Image.ANTIALIAS)
    pixels = partial.load()
    for x in range(partial.width):
        for y in range(partial.height):
            pixels[x, y] = 0 if pixels[x, y] == 255 else 1
    # partial.show()
    pixels = np.array(partial.getdata())
    pixels.shape = partial.width * partial.height
    return pixels


def crop_image(filepath):
    im = Image.open(filepath)
    data = list()
    data.append(crop_image_part(im, (0, 0, 26, 26)))
    data.append(crop_image_part(im, (2, 0, 28, 26)))
    data.append(crop_image_part(im, (0, 2, 26, 28)))
    data.append(crop_image_part(im, (2, 2, 28, 28)))
    return np.array(data)


def blur_gaussian():
    path1 = '/Users/Mar/PycharmProjects/BackPropagation/forREADME/regularization2.png'
    im = Image.open('TRAIN/1/0.bmp')
    pixels = im.load()
    for x in range(im.width):
        for y in range(im.height):
            pixels[x, y] = 0 if pixels[x, y] == 255 else 1
    im = im.filter(ImageFilter.SMOOTH)
    im.show()
    pixels = np.array(im.getdata())
    pixels.shape = (im.width, im.height)
    print(pixels)


def gaussian_blur_opencv(filepath):
    img = cv2.imread(filepath, 0)
    (width, height) = img.shape
    for x in range(width):
        for y in range(height):
            img[x, y] = 1 if img[x, y] == 255 else 0
    blur = cv2.GaussianBlur(img, (3, 3), 0)
    # print(blur)
    # plt.subplot(121), plt.imshow(img), plt.title('Original')
    # plt.xticks([]), plt.yticks([])
    # plt.subplot(122), plt.imshow(blur), plt.title('Blurred')
    # plt.xticks([]), plt.yticks([])
    # plt.show()
    blur.shape = width * height
    return 1 - blur


if __name__ == '__main__':
    (X, Y) = get_data(True)
    print(len(Y))

