from activation_functions import *
from layer import Layer_Dense
from regularization.dropout import Dropout


class NN:
    def __init__(self, l1, l2, input_size, hidden_sizes, output_size, 
                 hidden_activations: list[type[Activation]] | None = None, output_activation: Activation = Activation_Linear(), dropout_rates=None):
        self.layers = []
        prev_size = input_size
        
        # Default to ReLU if no activations specified
        activation_function = Activation_Linear
        if hidden_activations is None:
            hidden_activations = [activation_function for _ in hidden_sizes]
        
        # Default to no dropout
        if dropout_rates is None:
            dropout_rates = [0.0] * len(hidden_sizes)
            
        # Create hidden layers
        for size, activation, rate in zip(hidden_sizes, hidden_activations, dropout_rates):
            self.layers.append(Layer_Dense(prev_size, size, activation(), l1=l1, l2=l2))
            # if rate > 0:
            #     self.layers.append(Dropout(rate))
            prev_size = size
        
        # Output layer
        self.layers.append(Layer_Dense(prev_size, output_size, output_activation))
        
    def forward(self, inputs, training=True):
        for layer in self.layers:
            if isinstance(layer, Dropout):
                inputs = layer.forward(inputs, training)
            else:
                inputs = layer.forward(inputs)
        self.output = inputs
        return self.output

        