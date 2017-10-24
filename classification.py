import numpy as np
from preprocess import get_data
import random
import sys

minx, maxx = 0, 1  # range of input
miny, maxy = 0, 1  # range of output
learning_rate = 0.05
hidden_units = [100, 80, 50]  # each number represents the number of units of hidden layers
activation = ['tanH', 'tanH', 'tanH', 'softmax']  # activation function of each layer
regularization = [0,0.01,0.02,0.04,0.08,0.16,0.32,0.64,1.28,2.56,5.12,10.24]
max_epoch = 50000
momentum_strength = 0.5
batchSize = 256


def softmax(X):
    X = np.exp(X)
    sumX = np.sum(X, axis=1, keepdims=True)
    return X / sumX


activation_functions = {
    'sigmoid': (lambda x: 1/(1 + np.exp(-x)),
                      lambda x: x * (1 - x),  (0,  1), .45),
    'tanH':  (lambda x: np.tanh(x),
                      lambda x: 1 - x**2,     (0, -1), 0.005),
    'ReLU':  (lambda x: x * (x > 0),
              lambda x: x > 0,        (0, maxx), 0.0005),
    'softmax': (softmax, lambda x: x * (1 - x), (0,  1), .45)
}


def configure_algo(a):
    global activate, activatePrime, mina, maxa
    (activate, activatePrime, (mina, maxa), l) = activation_functions[a]  # L: learning rate, (mina,maxa): range of input


def random_weight(inputLayerSize, outputLayerSize):  # TODO
    inputLayerSize += 1  # add bias
    epsilon_init = np.sqrt(6) / np.sqrt(inputLayerSize + outputLayerSize)
    return np.random.uniform(low=-epsilon_init, high=epsilon_init, size=(inputLayerSize, outputLayerSize))


def random_weight2(inputLayerSize, outputLayerSize):
    inputLayerSize += 1
    return np.random.uniform(low=-0.01, high=0.01, size=(inputLayerSize, outputLayerSize))


def initialize_weight(X, Y):
    Theta = list()
    Theta.append(random_weight(X.shape[1], hidden_units[0]))
    for i in range(len(hidden_units)-1):
        Theta.append(random_weight(hidden_units[i], hidden_units[i+1]))
    Theta.append(random_weight(hidden_units[-1], Y.shape[1]))
    return Theta


def add_bias(X):
    return np.insert(X, obj=[0], values=[1], axis=1)


def remove_bias(X):
    return np.delete(X, obj=[0], axis=1)


def next_batch(X, Y, batchSize=256):
    for i in np.arange(0, X.shape[0], batchSize):
        yield (X[i:i + batchSize], Y[i:i + batchSize])


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


def epoch(mX, Y, Theta, reg_lambda, pre_Theta):
    # forward
    O = list()
    O.append(mX)
    i = 0
    for theta in Theta:
        configure_algo(activation[i])
        o = activate(O[i].dot(theta))
        O.append(add_bias(o))
        i += 1
    O[-1] = remove_bias(O[-1])
    # backward
    # ======momentum======
    momentum = (np.array(Theta) - np.array(pre_Theta)) * momentum_strength
    pre_Theta = Theta
    # ======momentum======
    mY = O[-1]
    E = Y-mY  # negative
    e = E
    step = learning_rate / mX.shape[0]
    for i in range(len(Theta)):
        # ======= Regularization ========
        forReg = - Theta[-1 - i] * step * reg_lambda
        forReg = np.delete(forReg, obj=[0], axis=0)  # bias should not be regularized
        forReg = np.insert(forReg, obj=[0], values=[0], axis=0)
        Theta[-1 - i] += forReg
        # ======= Regularization ========
        Theta[-1 - i] += step * O[-2 - i].T.dot(e) # hidden top layer * error top layer
        # for next layer
        if i == len(Theta) - 1:
            break
        configure_algo(activation[-2-i])
        e = e.dot(Theta[-1-i].T) * activatePrime(O[-2-i])  # next layer
        e = remove_bias(e)  # remove added bias from hidden layer
    # ======momentum======
    Theta += momentum
    return (mY, Theta, pre_Theta)


def train_model(reg_lambda, X, Y):
    global learning_rate, momentum_strength
    mX = add_bias(X)  # add bias before the first epoch
    pre_Theta = Theta = initialize_weight(X, Y)
    # print('initialize weights:')
    # print(Theta)
    min_avg_cost = 1e20
    i = 0
    j = 0
    k = 0
    init_learning_rate = learning_rate
    accuracy = 0
    try:
        while learning_rate > init_learning_rate / 512 and accuracy != 1 and k < max_epoch:
            # MY = np.array(np.zeros((1, Y.shape[1])))
            # for (Xb, Yb) in next_batch(mX, Y):
            (mY, Theta, pre_Theta) = epoch(mX, Y, Theta, reg_lambda, pre_Theta)
            #     MY = np.concatenate((MY, mY), axis=0)
            # MY = np.delete(MY, obj=[0], axis=0)
            j += 1
            k += 1
            # mY = np.array(MY)
            Cost = cross_entropy(Y, mY)
            avg_cost = np.sum(Cost)
            if avg_cost <= min_avg_cost:
                min_avg_cost = avg_cost
                i = 0
            else:
                i += 1
            if i == 30:
                learning_rate /= 2
                momentum_strength = 0.9
                i = 0
                accuracy = calculate_accuracy(Y, mY)
                print('accuracy of training set: %f' % accuracy)
                j = 0
            if j == 10000:  # keep track
                accuracy = calculate_accuracy(Y, mY)
                print('accuracy of training set: %f' % accuracy)
                j = 0
    except Exception or KeyboardInterrupt as e:
        print('accuracy of training set: %f' % accuracy)
        Theta = np.array(Theta)
        Theta.dump('classification.model')
        raise e
    # print('final model:')
    # print(Theta)
    return Theta


def predict(Theta, X, Y):
    mX = add_bias(X)
    O = list()
    O.append(mX)
    i = 0
    for theta in Theta:
        configure_algo(activation[i])
        o = activate(O[i].dot(theta))
        O.append(add_bias(o))
        i += 1
    O[-1] = remove_bias(O[-1])
    # evaluate results
    mY = O[-1]
    accuracy = calculate_accuracy(Y, mY)
    print('accuracy of test set: %f' % accuracy)
    return accuracy


def generate_batch(X, I, start, stop):
    X_test = list()
    X_train = list()
    stop = len(I) if len(I) < stop else stop
    for i in range(start, stop):
        X_test.append(X[I[i]])
    for i in range(start):
        X_train.append(X[I[i]])
    for i in range(stop, X.shape[0]):
        X_train.append(X[I[i]])
    return (np.array(X_test), np.array(X_train))


def cross_validation(reg_lambda, fold=5):
    (X, Y) = get_data()
    total = X.shape[0]
    I = np.arange(total)
    random.shuffle(I)
    random.shuffle(I)
    random.shuffle(I)
    batchSize = round(total / fold)
    start = 0
    accuracy = list()
    for i in range(fold):
        (X_test, X_train) = generate_batch(X, I, start, start + batchSize)
        (Y_test, Y_train) = generate_batch(Y, I, start, start + batchSize)
        Theta = train_model(reg_lambda, X_train, Y_train)
        accuracy.append(predict(Theta, X_test, Y_test))
        start += batchSize
    avg_accuracy = np.mean(accuracy)
    print('lambda:%f\navg_accuracy:%f' % (reg_lambda, avg_accuracy))
    print('detailed accuracy: ')
    print(accuracy)
    return avg_accuracy


if __name__ == '__main__':
    (X, Y) = get_data()  # modify the path inside
    Theta = train_model(10.24, X, Y)
    Theta = np.array(Theta)
    Theta.dump('classification.model')
    #
    # Theta = np.load('classification2.model')
    # print(Theta)
    # predict(Theta, X, Y)