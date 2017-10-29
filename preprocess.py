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


def get_data(reverse=False, crop=False, rotate_extend=False):
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
            im = Image.open('TRAIN/%d/' % (i + 1) + file)
            if crop:
                # crop in all direction
                X.append(crop_image_part(im, (0, 0, 26, 26)))
                X.append(crop_image_part(im, (2, 0, 28, 26)))
                X.append(crop_image_part(im, (0, 2, 26, 28)))
                X.append(crop_image_part(im, (2, 2, 28, 28)))
                for j in range(4):
                    y = np.zeros(14)
                    y[i] = 1
                    Y.append(y)
            if rotate_extend:
                X.append(rotate(im, 10))
                X.append(rotate(im, -10))
                for j in range(2):
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
    pixels = np.array(partial.getdata())
    pixels.shape = partial.width * partial.height
    return pixels


def rotate(im, angle):
    pixels = im.load()
    for x in range(im.width):
        for y in range(im.height):
            pixels[x, y] = 0 if pixels[x, y] == 255 else 1
    changed = im.rotate(angle)
    pixels = np.array(changed.getdata())
    pixels.shape = changed.width * changed.height
    return pixels


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


def training_set(crop=True, rotate_e=False):
    X = list()
    Y = list()
    for i in range(14):
        for k in range(205):
            x = image2array('TRAIN/%d/%d.bmp' % (i+1, k), reverse=True)
            X.append(x)
            y = np.zeros(14)
            y[i] = 1
            Y.append(y)
            im = Image.open('TRAIN/%d/%d.bmp' % (i+1, k))
            if crop:
                # crop in all direction
                X.append(crop_image_part(im, (0, 0, 26, 26)))
                X.append(crop_image_part(im, (2, 0, 28, 26)))
                X.append(crop_image_part(im, (0, 2, 26, 28)))
                X.append(crop_image_part(im, (2, 2, 28, 28)))
                for j in range(4):
                    y = np.zeros(14)
                    y[i] = 1
                    Y.append(y)
            if rotate_e:
                X.append(rotate(im, 10))
                X.append(rotate(im, -10))
                for j in range(2):
                    y = np.zeros(14)
                    y[i] = 1
                    Y.append(y)
    return (np.array(X), np.array(Y))

def test_set():
    X = list()
    Y = list()
    for i in range(14):
        for k in range(205, 256):
            x = image2array('TRAIN/%d/%d.bmp' % (i + 1, k), reverse=True)
            X.append(x)
            y = np.zeros(14)
            y[i] = 1
            Y.append(y)
    return (np.array(X), np.array(Y))


if __name__ == '__main__':
    im = Image.open('TRAIN/1/0.bmp')
    rotate(im, 0)
