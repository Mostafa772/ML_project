from abc import ABC, abstractmethod

import numpy as np


class Activation(ABC):
    @abstractmethod
    def forward(self, inputs):
        raise NotImplementedError

    @abstractmethod
    def backward(self, dvalues):
        raise NotImplementedError

    def __call__(self, inputs):
        return self.forward(inputs)

class Activation_Linear(Activation):
    """Linear activation for regression output"""
    def forward(self, inputs):
        return inputs

    def backward(self, dvalues):
        return 1

class Activation_ReLU(Activation):
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
        return self.output

    def backward(self, dvalues):
        # Since we need to modify the orginial variable,
        # let's make a copy of the values first
        self.dinputs = dvalues.copy()

        # zero gradient where input values were negative
        self.dinputs[self.inputs <= 0] = 0
        return self.dinputs
        

class Activation_Leaky_ReLU(Activation):
    def forward(self, inputs, alpha=0.01):
        self.inputs = inputs
        self.output = np.where(inputs > 0, inputs, alpha * inputs)
        return self.output

    def backward(self, dvalues, alpha=0.01):  # f′(x) = {1, α : if  x > 0 ; if x ≤ 0}
        self.dinputs = dvalues.copy()
        self.dinputs = np.where(self.inputs > 0, 1, alpha)
        self.dinputs *= dvalues
        return self.dinputs

class Activation_Sigmoid(Activation):
    def forward(self, inputs):
        self.inputs = inputs
        self.output = 1/(1 + np.exp(-inputs))
        return self.output

    def backward(self, dvalues):  # f′(x) = σ(x)⋅(1−σ(x))
        fx = self.forward(dvalues)
        return fx * (1 - fx)


class Activation_Tanh(Activation):
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.tanh(inputs)
        return self.output

    def backward(self, dvalues):  # f′(x) = 1−tanh**2(x)
        self.dinputs = dvalues.copy()
        self.dinputs = 1 - (self.output)**2
        self.dinputs *= dvalues
        return self.dinputs

class Activation_Softmax(Activation):
    def forward(self, inputs):
        self.inputs = inputs
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
        return probabilities

    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)

        for index, (single_output, single_dvalues) in \
                enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1, 1)

            jacobian_matrix = np.diagflat(single_output) - \
                                np.dot(single_output, single_output.T)

            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)
        return self.dinputs
