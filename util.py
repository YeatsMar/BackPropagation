import numpy as np

learning_rate = 0.05
max_error = 1e-4


def sigmoid(x):
    return 1 / ( 1 + np.exp(-x) )


def sigmoid_deviation(x):
    return sigmoid(x) * (1 - sigmoid(x))


def initialize_weight(inputLayerSize, outputLayerSize):
    epsilon_init = np.sqrt(6) / np.sqrt(inputLayerSize + outputLayerSize)
    return np.random.uniform(low=-epsilon_init, high=epsilon_init, size=(inputLayerSize, outputLayerSize))


X = np.array([[0,0], [0,1], [1,0], [1,1]])
Y = np.array([ [0],   [1],   [1],   [0]])


def ameliorate_input(X):
    return np.insert(X, obj=[0], values=[1], axis=1)

X = ameliorate_input(X)
hidden_units = 3 # array for more layers
W1 = initialize_weight(X.shape[1], hidden_units)
W2 = initialize_weight(hidden_units+1, Y.shape[1]) # loop


def epoch():
    global X, Y, W1, W2, Z
    H = sigmoid(X.dot(W1))
    H = ameliorate_input(H)
    Z = sigmoid(H.dot(W2))
    E = Y - Z
    dZ = E * sigmoid_deviation(Z) # element by element multiply
    W2 += H.T.dot(dZ)
    dH = dZ.dot(W2.T) * sigmoid_deviation(H)
    W1 += X.T.dot(dH)  # TODO: ignore added value





if __name__ == '__main__':
    for i in range(60000):
        epoch()
    print(Z)