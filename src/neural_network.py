import numpy as np
import random
import pandas as pd
from src.optimizers import *
from src.activation_functions import *
from src.utils import *
from src.model_regularization import *
from src.layer import *
from src.data_preprocessing import *
from src.batch_normalization import * 

class NN:
    def __init__(self, l1, l2, input_size, hidden_sizes, output_size,
                 hidden_activations=None, dropout_rates=None,
                 use_batch_norm=None, output_activation=Activation_Sigmoid()):
        self.layers = []
        prev_size = input_size

        # Default activations to ReLU
        if hidden_activations is None:
            hidden_activations = [Activation_ReLU() for _ in hidden_sizes]

        # Default dropout rates to 0
        if dropout_rates is None:
            dropout_rates = [0.0] * len(hidden_sizes)

        # Default batch_norm to False for all layers
        if use_batch_norm is None:
            use_batch_norm = [False] * len(hidden_sizes)
        else:
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
        # self.layers.append(output_activation)

    def forward(self, inputs, training=True):
        for layer in self.layers:
            # Pass training flag to relevant layers
            if isinstance(layer, (Dropout, BatchNormalization)):
                layer.forward(inputs, training=training)
            else:
                layer.forward(inputs)
            inputs = layer.output
        self.output = inputs
        return self.output
