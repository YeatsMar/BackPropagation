import numpy as np

max_error = 1e-2

minx, maxx = -3.14, 3.14
miny, maxy = -1, 1
numx = int(maxx*5+1)

def f(x): return np.sin(x)

X = np.linspace(minx, maxx, num=numx)
X.shape = (numx, 1)
Y = f(X)

hidden_units = [5,2] # array for more layers
activation = 'sigmoid'

activation_functions = {
    'sigmoid': (lambda x: 1/(1 + np.exp(-x)),
                      lambda x: x * (1 - x),  (0,  1), .45),
    'tanh':  (lambda x: np.tanh(x),
                      lambda x: 1 - x**2,     (0, -1), 0.005),
    'ReLU':  (lambda x: x * (x > 0),
              lambda x: x > 0,        (0, maxx), 0.0005),
    #TODO: softmax
}

(activate, activatePrime, (mina, maxa), learning_rate) = activation_functions[activation]  # L: learning rate, (mina,maxa): range of input



def normalize_y(Y):
    return (Y - miny)*(maxa - mina)/(maxy - miny) + mina

def normalize_x(X):
    pass  # TODO


def random_weight(inputLayerSize, outputLayerSize):
    inputLayerSize += 1  # add bias
    epsilon_init = np.sqrt(6) / np.sqrt(inputLayerSize + outputLayerSize)
    return np.random.uniform(low=-epsilon_init, high=epsilon_init, size=(inputLayerSize, outputLayerSize))


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
Y = normalize_y(Y)
initialize_weight()

def epoch():
    global mX, Y, Theta, mY  # Theta is weight
    # forward
    O = list()
    O.append(mX)
    i = 0
    for theta in Theta:
        o = activate(O[i].dot(theta))
        O.append(add_bias(o))
        i += 1
    O[-1]=remove_bias(O[-1])
    # backward
    mY = O[-1]
    E = Y - mY  # negative
    e_top = E * activatePrime(mY)
    Theta[-1] += learning_rate/X.shape[0] * O[-2].T.dot(e_top)  # hidden top layer * error top layer
    e_pre = e_top
    for i in range(1, len(Theta)):
        e = e_pre.dot(Theta[-i].T) * activatePrime(O[-1-i])  # next layer
        e = remove_bias(e)  # remove added bias from hidden layer
        Theta[-1-i] += learning_rate/X.shape[0] * O[-2-i].T.dot(e)
        e_pre = e


if __name__ == '__main__':
    print('initialize weights:')
    print(Theta)
    min_avg_cost = 10
    i = 0
    init_learning_rate = learning_rate
    while learning_rate > init_learning_rate / 1024: #todo: early stopping
        epoch()
        Cost = np.abs(Y-mY)
        avg_cost = np.mean(Cost)
        if avg_cost <= min_avg_cost:
            min_avg_cost = avg_cost
            i = 0
        else:
            i += 1
        if i == 10:  # todo: learning decay
            learning_rate /= 2
            i = 0
            print('error: %f' % avg_cost)
            print('=====Result====')
            for i in range(X.shape[0]):
                print(Y[i], mY[i])
    print('final model:')
    print(Theta)