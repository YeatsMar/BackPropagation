from PIL import Image
import numpy as np
import os

def image2array(filepath):
    im = Image.open(filepath)
    print(im.format, im.size, im.mode)
    pixels = im.load()
    for x in range(im.width):
        for y in range(im.height):
            pixels[x, y] = 1 if pixels[x, y] == 255 else 0
    pixels = np.array(im.getdata())
    pixels.shape = im.width * im.height
    return pixels

def get_data():
    X = list()
    Y = list()
    for i in range(14):
        for file in os.listdir('TRAIN/%d'%(i+1)):
            if '.bmp' not in file:
                continue
            x = image2array('TRAIN/%d/'%(i+1) + file)
            X.append(x)
            y = np.zeros(14)
            y[i] = 1
            Y.append(y)
    return (X, Y)

if __name__ == '__main__':
    (X, Y) = get_data()