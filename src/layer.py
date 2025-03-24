from abc import ABC, abstractmethod

import numpy as np

from activation_functions import Activation, Activation_Sigmoid


class Layer(ABC):
    def __init__(self,n_inputs: int, n_neurons: int, activation: Activation) -> None:
        self.activation = activation

        scale = np.sqrt(2/n_inputs) # For Tanh/Sigmoid activation functions
        self.weights: np.ndarray = np.random.randn(n_inputs, n_neurons) * scale     # Random initialization
        self.biases: np.ndarray = np.zeros((1, n_neurons))

        self.weight_momentums: np.ndarray = np.zeros_like(self.weights)
        self.weight_cache: np.ndarray = np.zeros_like(self.weights)
        self.bias_momentums: np.ndarray = np.zeros_like(self.biases)
        self.bias_cache: np.ndarray = np.zeros_like(self.biases)

        self.dweights: np.ndarray
        self.dbiases: np.ndarray

    @abstractmethod
    def forward(self, inputs: np.ndarray):
        raise NotImplemented

    @abstractmethod
    def backward(self, dvalues: np.ndarray):
        raise NotImplemented

class Layer_Dense(Layer):
    def __init__(self, n_inputs: int, n_neurons: int, activation: Activation = Activation_Sigmoid(), l1: float = 0, l2: float = 0):
        super().__init__(n_inputs, n_neurons, activation)
        self.l1 = l1
        self.l2 = l2
        
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.inputs = inputs
        self.net = np.dot(inputs, self.weights) + self.biases
        self.output = self.activation(self.net)
        return self.output

    def backward(self, dvalues: np.ndarray) -> np.ndarray:
        # Compute gradient of the activation function
        derivative = self.activation.backward(self.net)
        dvalues *= derivative
        
        # Gradients on weights and biases
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dweights = np.dot(self.inputs.T, dvalues)
        
        # # L1 regularization on weights
        # if self.l1 > 0:
        #     self.dweights += self.l1 * np.sum(np.linalg.norm(self.weights, ord=1))
        
        # # L2 regularization on weights
        # if self.l2 > 0:
        #     self.dweights += 2 * self.l2 * np.sum(np.square(self.weights))
        
        # Gradient on inputs
        self.dinputs = np.dot(dvalues.T, self.inputs)
        
        return self.dinputs
