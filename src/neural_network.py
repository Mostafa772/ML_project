import random

import numpy as np
import pandas as pd

from src.activation_functions import *
from src.batch_normalization import *
from src.data_preprocessing import *
from src.layer import *
from src.dropout import Dropout
from src.optimizers import *
from src.utils import *

class Base_NN:
    def __init__(self):
        self.layers = []
    
    def forward(self, inputs: np.ndarray, training: bool = False) -> np.ndarray:
        for layer in self.layers:
            layer.forward(inputs)
            inputs = layer.output
        self.output = inputs
        return self.output

class NN(Base_NN):
    def __init__(self, l1, l2, input_size, hidden_sizes, output_size,
                 hidden_activations=None, dropout_rates=None,
                 use_batch_norm=None, output_activation=Activation_Sigmoid()):
        super().__init__()
        prev_size = input_size

        # Default activations to ReLU
        if hidden_activations is None:
            hidden_activations = [Activation_ReLU() for _ in hidden_sizes]

        # Default dropout rates to 0
        if dropout_rates is None:
            dropout_rates = [0.0] * len(hidden_sizes)

        # Default batch_norm to False for all layers
        if isinstance(use_batch_norm, bool):
            use_batch_norm = [use_batch_norm] * len(hidden_sizes)
        assert len(use_batch_norm) == len(hidden_sizes), \
            "use_batch_norm must have the same length as hidden_sizes"
        print(hidden_sizes, hidden_activations, dropout_rates, use_batch_norm)
        # Create hidden layers
        for size, activation, rate, bn_flag in zip(hidden_sizes, hidden_activations,
                                                   dropout_rates, use_batch_norm):
            # Add dense layer
            self.layers.append(Layer_Dense(prev_size, size, l1=l1, l2=l2))

            # Add batch normalization if specified
            if bn_flag:
                self.layers.append(BatchNormalization())

            # Add activation
            self.layers.append(activation())

            # Add dropout if rate > 0
            if rate > 0:
                self.layers.append(Dropout(rate))

            prev_size = size

        # Output layer
        # initialize the output layer with the same weight initialization method as the hidden layers
        # self.layers.append(Layer_Dense(prev_size, output_size, weights_init=weight_init))
        self.layers.append(Layer_Dense(prev_size, output_size))
        self.layers.append(output_activation)

    def forward(self, inputs, training=True) -> np.ndarray:
        for layer in self.layers:
            # Pass training flag to relevant layers
            if isinstance(layer, (Dropout, BatchNormalization)):
                layer.forward(inputs, training=training)
            else:
                layer.forward(inputs)
            inputs = layer.output
        self.output = inputs
        return self.output
