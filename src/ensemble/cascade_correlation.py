import numpy as np

from src.layer import Layer_Dense
from src.neural_network import Base_NN


class CascadeCorrelation(Base_NN):
    def __init__(self, input_size, output_size, activation, output_activation, max_depth = 10):
        super().__init__()
        self.depth = 1
        self.max_depth = max_depth
        self.activation = activation
        self.input_size = input_size

        self.layers = [CascadeCorrelationLayer(input_size, 1), activation(), CascadeCorrelationLayer(input_size + 1, output_size, is_output=True), output_activation()]

    def add_neuron(self) -> 'CascadeCorrelation':
        if self.is_limit_reached():
            return self

        out_act = self.layers[-1]
        out_layer = self.layers[-2]
        last_hidden_neuron = self.layers[-4]

        # freeze previous neuron training
        last_hidden_neuron.is_frozen = True

        # add an input to the output layer so that it can accept the new neuron input
        out_layer.add_input_connection()

        # create new neuron
        new_neuron = CascadeCorrelationLayer(self.depth + self.input_size , 1)
        self.depth += 1

        # add neuron after all the others
            # substituting to output ensures it is at last position
        self.layers[-2] = new_neuron
        self.layers[-1] = self.activation()
            # output is added at the very end
        self.layers.append(out_layer)
        self.layers.append(out_act)

        return self

    def is_limit_reached(self) -> bool:
        """
        To use externally, returns true when the depth limit of the network has been reached
        """
        return self.depth >= self.max_depth

class CascadeCorrelationLayer(Layer_Dense):
    def __init__(self, n_inputs: int, n_outputs: int, is_output = False):
        super().__init__(n_inputs=n_inputs, n_neurons=n_outputs)
        self.is_frozen = False
        self.is_output = is_output

    def add_input_connection(self):
        prev = self.weights.shape
        scale = np.sqrt(2 / (self.weights.shape[0] + 1)) / 10
        new_weights = np.random.randn(1, 1) * scale

        assert self.weights.shape[1] == new_weights.shape[1]
        self.weights = np.append(self.weights, new_weights, axis=0) # add row
        if hasattr(self, 'weight_cache'):
            zero_padding = np.zeros((1,1))
            self.weight_momentums = np.append(self.weight_momentums, zero_padding, axis=0)
            self.weight_cache = np.append(self.weight_cache, zero_padding, axis=0)

    def forward(self, inputs: np.ndarray):
        """
        Fowards the output of the neuron with the input
        """
        self.net = super().forward(inputs)
        if self.is_output:
            self.output = self.net
        else:
            self.output = np.concatenate([self.net, inputs], axis=1)

        return self.output

    def backward(self, dvalues: np.ndarray):
        if self.is_frozen:
            self.dinputs = np.array(0)
            self.dweights = np.array(0)
            self.dbiases = np.array(0)
            return np.zeros_like((dvalues.shape[0], self.weights.shape[0]))
        
        dvalues = dvalues[:, :self.weights.shape[1]]
        
        back = super().backward(dvalues)

        assert self.weights.shape == self.dweights.shape, f"weights {self.weights.shape}, dw {self.dweights.shape}"
        return back