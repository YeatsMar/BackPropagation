import random
import numpy as np
from classification import generate_batch
from preprocess import get_data


class CnnData:
    cvIndex = 0
    batchIndex = 0

    def __init__(self, fold=5, rotate=False, crop=False):
        self.fold = fold
        (self.X, self.Y) = get_data(reverse=True, crop=crop, rotate_extend=rotate)
        self.sample_num = self.X.shape[0]
        self.blockSize = round(self.sample_num / self.fold)
        self.I = np.arange(self.sample_num)
        for i in range(3):
            random.shuffle(self.I)

    def nextCVRound(self):
        (self.X_test, self.X_train) = generate_batch(self.X, self.I, self.cvIndex, self.cvIndex + self.blockSize)
        (self.Y_test, self.Y_train) = generate_batch(self.Y, self.I, self.cvIndex, self.cvIndex + self.blockSize)
        self.cvIndex += self.blockSize

