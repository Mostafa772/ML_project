import numpy as np

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
    def __init__(self, l1, l2, input_size, hidden_size: list[int] | int, output_size,
                 hidden_activation=Activation_ReLU, dropout_rate=0.0,
                 use_batch_norm=False, output_activation=Activation_Sigmoid(), weights_init: str = 'gaussian', n_h_layers: int = 1):
        super().__init__()
        prev_size = input_size

        # Create hidden layers

        if isinstance(hidden_size, int):
            hidden_size = [hidden_size for _ in range(n_h_layers)]

        for size in hidden_size:
            # Add dense layer
            self.layers.append(Layer_Dense(prev_size, size, l1=l1, l2=l2, weights_init=weights_init))

            # Add batch normalization if specified
            if use_batch_norm:
                self.layers.append(BatchNormalization())

            # Add activation
            self.layers.append(hidden_activation())

            # Add dropout if rate > 0
            if dropout_rate > 0:
                self.layers.append(Dropout(dropout_rate))

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
