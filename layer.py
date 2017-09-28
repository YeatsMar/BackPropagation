import util

class Layer:
    input = list()
    output = list()
    hidden_unit = 10 # default

    def __init__(self, input, output, hidden_unit=10):
        self.input = input
        self.output = output
        self.hidden_unit = hidden_unit


if __name__ == '__main__':
    pass