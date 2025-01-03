import numpy as np
import pandas as pd
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        # the random function takes the shape of the matrix
        self.weights = 0.10*np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        # an output variable for our layer.
        self.output = np.dot(inputs, self.weights) + self.biases


class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs) #This compares the values passed(inputs to zero). If greater than, then keep it. else make it zero
    def backward():
        pass


class Activation_Leaky_ReLU:
    def forward(self, inputs, alpha=0.01):
        self.inputs = inputs
        self.output = np.where(inputs > 0, inputs, alpha * inputs)

    def backward(self, dvalues):  # f′(x) = {1, α : if  x > 0 ; if x ≤ 0
        self.dinputs = np.where(self.inputs > 0, 1, alpha)
        self.dinputs *= dvalues


class Activation_Sigmoid:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = 1/(1 + np.exp(-inputs))

    def backward(self, dvalues):  # f′(x) = σ(x)⋅(1−σ(x))
        self.dinputs = self.output * (1 - self.output)
        self.dinputs *= dvalues


class Activation_Tanh:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.tanh(inputs)

    def backward(self, dvalues):  # f′(x) = 1−tanh**2(x)
        self.dinputs = 1 - (self.output)**2
        self.dinputs *= dvalues


data = pd.read_csv('./data/Monk_1/monks-1.train', header=None, sep='\s+')
X = data.iloc[1:, :-1].to_numpy()

layer1 = Layer_Dense(X.shape[1], 5)

activation1 = Activation_ReLU()
activation2 = Activation_Leaky_ReLU()
activation3 = Activation_Sigmoid()
activation4 = Activation_Tanh()

layer1.forward(X)

# Activation function ReLU
activation1.forward(layer1.output)

# Activation function Leaky_ReLU
# activation2.forward(layer1.output)

# Activation function Sigmoid
# activation3.forward(layer1.output)

# Activation function Tanh
# activation4.forward(layer1.output)

# print(layer1.output)
print(activation4.output)