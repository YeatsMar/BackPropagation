import random
import numpy as np
from classification import generate_batch
from preprocess import get_data


class CnnData:
    (X, Y) = get_data(reverse=True)
    fold = 5
    cvIndex = 0
    sample_num = X.shape[0]
    blockSize = round(sample_num / fold)
    I = np.arange(sample_num)
    batchIndex = 0

    def __init__(self):
        for i in range(3):
            random.shuffle(self.I)

    def nextCVRound(self):
        (self.X_test, self.X_train) = generate_batch(self.X, self.I, self.cvIndex, self.cvIndex + self.blockSize)
        (self.Y_test, self.Y_train) = generate_batch(self.Y, self.I, self.cvIndex, self.cvIndex + self.blockSize)
        self.cvIndex += self.blockSize

