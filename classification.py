import numpy as np
from preprocess import get_data

minx, maxx = 0, 1
miny, maxy = 0, 1

activation = 'softmax'

def softmax(X):
    X = np.exp(X)
    sumX = np.sum(X, axis=1, keepdims=True)
    # sumX = np.repeat(sumX, X.shape[1], axis=1)
    return X / sumX

regularization = [0,0.01,0.02,0.04,0.08,0.16,0.32,0.64,1.28,2.56,5.12,10.24]

reg_lambda = regularization[0]

activation_functions = {
    'sigmoid': (lambda x: 1/(1 + np.exp(-x)),
                      lambda x: x * (1 - x),  (0,  1), .45),
    'tanh':  (lambda x: np.tanh(x),
                      lambda x: 1 - x**2,     (0, -1), 0.005),
    'ReLU':  (lambda x: x * (x > 0),
              lambda x: x > 0,        (0, maxx), 0.0005),
    'softmax': (softmax, lambda x: x * (1 - x), (0,  1), .45)
}

# X = np.array([[0,0], [0,1], [1,0], [1,1]])
# Y = np.array([ [0],   [1],   [1],   [0]])
hidden_units = [100, 80, 50] # array for more layers

(X, Y) = get_data()

(activate, activatePrime, (mina, maxa), learning_rate) = activation_functions[activation]  # L: learning rate, (mina,maxa): range of input


def random_weight(inputLayerSize, outputLayerSize):  # TODO
    inputLayerSize += 1  # add bias
    epsilon_init = np.sqrt(6) / np.sqrt(inputLayerSize + outputLayerSize)
    return np.random.uniform(low=-epsilon_init, high=epsilon_init, size=(inputLayerSize, outputLayerSize))


def random_weight2(inputLayerSize, outputLayerSize):
    inputLayerSize += 1
    return np.random.uniform(low=-0.01, high=0.01, size=(inputLayerSize, outputLayerSize))


def initialize_weight():
    global Theta
    Theta = list()
    Theta.append(random_weight(X.shape[1], hidden_units[0]))
    for i in range(len(hidden_units)-1):
        Theta.append(random_weight(hidden_units[i], hidden_units[i+1]))
    Theta.append(random_weight(hidden_units[-1], Y.shape[1]))


def add_bias(X):
    return np.insert(X, obj=[0], values=[1], axis=1)


def remove_bias(X):
    return np.delete(X, obj=[0], axis=1)


mX = add_bias(X)  # add bias before the first epoch
initialize_weight()

def epoch():
    global mX, Y, Theta, mY  # Theta is weight
    # forward
    O = list()
    O.append(mX)
    i = 0
    for theta in Theta:
        o = activate(O[i].dot(theta))  # todo: only the last layer use softmax
        O.append(add_bias(o))
        i += 1
    O[-1] = remove_bias(O[-1])
    # backward
    mY = O[-1]
    E = Y-mY  # negative
    e_top = E * activatePrime(mY)  # element by element multiply
    # ======= Regularization ========
    forReg = - Theta[-1] * learning_rate / X.shape[0] * reg_lambda
    forReg = np.delete(forReg, obj=[0], axis=0)
    forReg = np.insert(forReg, obj=[0], values=[0], axis=0)
    Theta[-1] += forReg
    # ======= Regularization ========
    Theta[-1] += learning_rate/X.shape[0] * O[-2].T.dot(e_top)  # hidden top layer * error top layer
    e_pre = e_top
    for i in range(1, len(Theta)):
        e = e_pre.dot(Theta[-i].T) * activatePrime(O[-1-i])  # next layer
        e = remove_bias(e)  # remove added bias from hidden layer
        # ======= Regularization ========
        forReg = - Theta[-1-i] * learning_rate / X.shape[0] * reg_lambda
        forReg = np.delete(forReg, obj=[0], axis=0)
        forReg = np.insert(forReg, obj=[0], values=[0], axis=0)
        Theta[-1-i] += forReg
        # ======= Regularization ========
        Theta[-1-i] += learning_rate/X.shape[0] * O[-2-i].T.dot(e)
        e_pre = e


def cross_entropy(y, y2):
    return - y * np.log(y2) - (1-y)* np.log(1-y2)

def calculate_accuracy(Y, mY):
    correct_category = Y.argmax(axis=1)
    predicted_category = mY.argmax(axis=1)
    total = Y.shape[0]
    count = 0
    for i in range(total):
        if correct_category[i] == predicted_category[i]:
            count += 1
    return count/total


if __name__ == '__main__':
    X = np.array([[1, 2, 3, 4], [10, 11, 12,13]])
    print(softmax(X))
    # print('initialize weights:')
    # print(Theta)
    # min_avg_cost = 1e10
    # i = 0
    # init_learning_rate = learning_rate
    # while learning_rate > init_learning_rate / 1024:  # todo: early stopping
    #     epoch()
    #     Cost = cross_entropy(Y, mY)
    #     avg_cost = np.sum(Cost)
    #     if avg_cost <= min_avg_cost:
    #         min_avg_cost = avg_cost
    #         i = 0
    #     else:
    #         i += 1
    #     if i == 10:  # todo: learning decay
    #         learning_rate /= 2
    #         i = 0
    #         print('accuracy: %f' % calculate_accuracy(Y, mY))
    # print('final model:')
    # print(Theta)