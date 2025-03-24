import numpy as np

from layer import Layer


class Dropout(Layer):
    def __init__(self, rate: float):
        """
        Initialize a dropout layer.
        
        Parameters:
        - rate: Dropout rate (fraction of inputs to drop)
        """
        self.rate = 1 - rate  # Store keep rate instead of drop rate
        self.mask = None
        self.inputs = None
        self.output = None
        self.dinputs = None
        
    def forward(self, inputs, training=True):
        """
        Perform the forward pass with dropout.
        
        Parameters:
        - inputs: Input data
        - training: Boolean indicating training mode
        """
        self.inputs = inputs
        
        if training:
            self.mask = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate
            self.output = inputs * self.mask
        else:
            self.output = inputs

        return self.output
            
    def backward(self, dvalues):
        """
        Perform the backward pass through dropout.
        
        Parameters:
        - dvalues: Gradient of the loss with respect to the output
        """
        self.dinputs = dvalues * self.mask
        return self.dinputs