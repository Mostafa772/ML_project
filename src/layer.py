from abc import abstractmethod, ABC
import numpy as np
from numpy.typing import ArrayLike
import pandas as pd

from activation_functions import Activation, Activation_Linear

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
    def __init__(self, n_inputs: int, n_neurons: int, activation: Activation = Activation_Linear(), l1: float = 0.0, l2: float = 0.0):
        super().__init__(n_inputs, n_neurons, activation)
        self.l1 = l1
        self.l2 = l2
        
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.inputs = inputs
        net = np.dot(inputs, self.weights) + self.biases
        self.output = self.activation(net)
        return self.output

    def backward(self, dvalues: np.ndarray) -> np.ndarray:
        # Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        # L1 regularization
        if self.l1 > 0:
            dl1 = np.ones_like(self.weights)
            dl1[self.weights < 0] = -1
            self.dweights += self.l1 * dl1

        # L2 regularization
        if self.l2 > 0:
            self.dweights += 2 * self.l2 * self.weights

        # Gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)
        return self.dinputs

    def get_regularization_loss(self):
        """
        Calculate regularization loss for the layer.

        Returns:
        - regularization_loss: Combined L1 and L2 regularization loss
        """
        regularization_loss = 0

        if self.l1 > 0:
            regularization_loss += self.l1 * np.sum(np.abs(self.weights))

        if self.l2 > 0:
            regularization_loss += self.l2 * np.sum(self.weights * self.weights)

        return regularization_loss
