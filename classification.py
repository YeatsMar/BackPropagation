import numpy as np

learning_rate = 0.05
max_error = 1e-4
activation_functions = {
    'sigmoid': {

    }
}

X = np.array([[0,0], [0,1], [1,0], [1,1]])
Y = np.array([ [0],   [1],   [1],   [0]])
hidden_units = [2] # array for more layers



def sigmoid(x):
    return 1 / ( 1 + np.exp(-x) )


def sigmoid_deviation(x):
    return sigmoid(x) * (1 - sigmoid(x))


def random_weight(inputLayerSize, outputLayerSize):
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
        o = sigmoid(O[i].dot(theta))
        O.append(add_bias(o))
        i += 1
    O[-1] = np.delete(O[-1], [0], axis=1)
    # backward
    mY = O[-1]
    E = Y-mY  # negative
    e_top = E * sigmoid_deviation(mY)  # element by element multiply
    Theta[-1] += O[-2].T.dot(e_top)  # hidden top layer * error top layer
    e_pre = e_top
    for i in range(1, len(Theta)):
        e = e_pre.dot(Theta[-i].T) * sigmoid_deviation(O[-1-i])  # next layer
        e = remove_bias(e)  # remove added bias from hidden layer
        Theta[-1-i] += learning_rate * O[-2-i].T.dot(e)
        e_pre = e


if __name__ == '__main__':
    for i in range(60000):
        epoch()
    print(mY)
    print(random_weight(2,5))