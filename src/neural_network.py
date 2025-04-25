import numpy as np
import random
import pandas as pd
from itertools import product
from sklearn.model_selection import train_test_split
from src.optimizers import *
from src.activation_functions import *
from src.utils import *
from src.model_regularization import *
from src.layer import *
from src.data_preprocessing import *

class NN:
    def __init__(self, l1, l2, input_size, hidden_sizes, output_size,
                 hidden_activations=None, dropout_rates=None):
        self.layers = []
        prev_size = input_size

        # Default to ReLU if no activations specified
        if hidden_activations is None:
            hidden_activations = [Activation_ReLU() for _ in hidden_sizes]

        # Default to no dropout
        if dropout_rates is None:
            dropout_rates = [0.0] * len(hidden_sizes)

        # Create hidden layers
        for size, activation, rate in zip(hidden_sizes, hidden_activations, dropout_rates):
            self.layers.append(Layer_Dense(prev_size, size, l1=l1, l2=l2))
            self.layers.append(activation())
            if rate > 0:
                self.layers.append(Dropout(rate))
            prev_size = size

        # Output layer (no activation)
        self.layers.append(Layer_Dense(prev_size, output_size))

    def forward(self, inputs, training=True):
        for layer in self.layers:
            if isinstance(layer, Dropout):
                layer.forward(inputs, training)
            else:
                layer.forward(inputs)
            inputs = layer.output
        self.output = inputs
