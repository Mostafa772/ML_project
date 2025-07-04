import numpy as np
import pandas as pd

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons, weights_init="gaussian", l1=0.0, l2=0.0):
        
        if weights_init == "gaussian":
            self.weights = np.random.normal(0.0, 1.0, (n_inputs, n_neurons))  # std=1
        elif weights_init == "gaussian_scaled":
            self.weights = np.random.normal(0.0, 0.05, (n_inputs, n_neurons))  # smaller std
        elif weights_init == "xavier":
            scale = np.sqrt(2 / (n_inputs + n_neurons))
            self.weights = np.random.randn(n_inputs, n_neurons) * scale
        elif weights_init == "he":
            scale = np.sqrt(2 / n_inputs)
            self.weights = np.random.randn(n_inputs, n_neurons) * scale
        elif weights_init == "random":
            self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        else:
            raise ValueError(f"Unknown weights_init: {weights_init}")
        self.biases = np.zeros((1, n_neurons))
        self.l1 = l1
        self.l2 = l2
        self.dweights = None

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
        return self.output
   
    def backward(self, dvalues):
        # Gradients on parameters
        if isinstance(dvalues, (pd.DataFrame, pd.Series)):
            dvalues = dvalues.values

        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dweights = np.clip(self.dweights, -5.0, 5.0)
        self.dbiases = np.clip(self.dbiases, -5.0, 5.0)
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
