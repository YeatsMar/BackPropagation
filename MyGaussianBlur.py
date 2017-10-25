import math
import numpy as np
from PIL import Image


class MyGaussianBlur():
    # 初始化
    def __init__(self, radius=1, sigema=1.5):
        self.radius = radius
        self.sigema = sigema
        # 高斯的计算公式

    def calc(self, x, y):
        res1 = 1 / (2 * math.pi * self.sigema * self.sigema)
        res2 = math.exp(-(x * x + y * y) / (2 * self.sigema * self.sigema))
        return res1 * res2
        # 得到滤波模版

    def template(self):
        sideLength = self.radius * 2 + 1
        result = np.zeros((sideLength, sideLength))
        for i in range(sideLength):
            for j in range(sideLength):
                result[i, j] = self.calc(i - self.radius, j - self.radius)
        all = result.sum()
        return result / all
        # 滤波函数

    def filter(self, image, template):
        arr = np.array(image)
        height = arr.shape[0]
        width = arr.shape[1]
        newData = np.zeros((height, width))
        for i in range(self.radius, height - self.radius):
            for j in range(self.radius, width - self.radius):
                t = arr[i - self.radius:i + self.radius + 1, j - self.radius:j + self.radius + 1]
                a = np.multiply(t, template)
                newData[i, j] = a.sum()
        newImage = Image.fromarray(newData)
        return newImage


r = 1  # 模版半径，自己自由调整
s = 2  # sigema数值，自己自由调整
GBlur = MyGaussianBlur(radius=0)  # 声明高斯模糊类
temp = GBlur.template()  # 得到滤波模版
im = Image.open('/Users/Mar/PycharmProjects/BackPropagation/forREADME/regularization2.png')  # 打开图片
image = GBlur.filter(im, temp)  # 高斯模糊滤波，得到新的图片
image.show()  # 图片显示