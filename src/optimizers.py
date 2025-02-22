import numpy as np

from src.layer import Layer

class Optimizer_Base:
    """Base class for all optimizers"""
    def __init__(self, learning_rate=1.0, decay=0.0):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0

    def pre_update_params(self):
        """Update learning rate based on decay schedule"""
        if self.decay > 0:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

    def post_update_params(self):
        """Increment iteration counter"""
        self.iterations += 1

    def update_params(self, layer):
        """To be implemented by child classes"""
        raise NotImplementedError


class Optimizer_SGD(Optimizer_Base):
    """Stochastic Gradient Descent optimizer with momentum support"""
    def __init__(self, learning_rate=1.0, decay=0.0, momentum=0.0):
        super().__init__(learning_rate, decay)
        self.momentum = momentum

    def update_params(self, layer):
        if self.momentum:
            # Initialize momentum arrays if they don't exist
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)

            # Update with momentum
            weight_updates = self.momentum * layer.weight_momentums - \
                           self.current_learning_rate * layer.dweights
            layer.weight_momentums = weight_updates

            bias_updates = self.momentum * layer.bias_momentums - \
                         self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_updates
        else:
            # Vanilla SGD
            weight_updates = -self.current_learning_rate * layer.dweights
            bias_updates = -self.current_learning_rate * layer.dbiases

        layer.weights += weight_updates
        layer.biases += bias_updates


class Optimizer_Adagrad(Optimizer_Base):
    """Adaptive Gradient optimizer"""
    def __init__(self, learning_rate=1.0, decay=0.0, epsilon=1e-7):
        super().__init__(learning_rate, decay)
        self.epsilon = epsilon

    def update_params(self, layer):
        # Initialize accumulated gradients if they don't exist
        if not hasattr(layer, 'weight_accumulate'):
            layer.weight_accumulate = np.zeros_like(layer.weights)
            layer.bias_accumulate = np.zeros_like(layer.biases)

        # Update accumulated gradients
        layer.weight_accumulate += layer.dweights**2
        layer.bias_accumulate += layer.dbiases**2

        # Update parameters
        layer.weights += -self.current_learning_rate * layer.dweights / \
                        (np.sqrt(layer.weight_accumulate) + self.epsilon)
        layer.biases += -self.current_learning_rate * layer.dbiases / \
                       (np.sqrt(layer.bias_accumulate) + self.epsilon)


class Optimizer_RMSprop(Optimizer_Base):
    """RMSprop optimizer"""
    def __init__(self, learning_rate=0.001, decay=0.0, epsilon=1e-7, rho=0.9):
        super().__init__(learning_rate, decay)
        self.epsilon = epsilon
        self.rho = rho
        self.squared_gradients = {}

    def update_params(self, layer):
        layer_id = id(layer)
        
        # Initialize squared gradients if they don't exist
        if layer_id not in self.squared_gradients:
            self.squared_gradients[layer_id] = {
                'weights': np.zeros_like(layer.weights),
                'biases': np.zeros_like(layer.biases)
            }

        # Update squared gradients
        self.squared_gradients[layer_id]['weights'] = \
            self.rho * self.squared_gradients[layer_id]['weights'] + \
            (1 - self.rho) * layer.dweights**2
        self.squared_gradients[layer_id]['biases'] = \
            self.rho * self.squared_gradients[layer_id]['biases'] + \
            (1 - self.rho) * layer.dbiases**2

        # Update parameters
        layer.weights += -self.current_learning_rate * layer.dweights / \
                        (np.sqrt(self.squared_gradients[layer_id]['weights']) + self.epsilon)
        layer.biases += -self.current_learning_rate * layer.dbiases / \
                       (np.sqrt(self.squared_gradients[layer_id]['biases']) + self.epsilon)


class Optimizer_Adam(Optimizer_Base):
    """Adam optimizer"""
    def __init__(self, learning_rate=0.001, decay=0.0, epsilon=1e-7,
                 beta_1=0.9, beta_2=0.999):
        super().__init__(learning_rate, decay)
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def update_params(self, layer: Layer):

        # Update momentum with current gradients
        layer.weight_momentums = self.beta_1 * layer.weight_momentums +  (1 - self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1 - self.beta_1) * layer.dbiases

        # Get corrected momentum
        weight_momentums_corrected = layer.weight_momentums / (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / (1 - self.beta_1 ** (self.iterations + 1))

        # Update cache with squared current gradients
        layer.weight_cache = self.beta_2 * layer.weight_cache + (1 - self.beta_2) * layer.dweights**2
        layer.bias_cache = self.beta_2 * layer.bias_cache + (1 - self.beta_2) * layer.dbiases**2

        # Get corrected cache
        weight_cache_corrected = layer.weight_cache / (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / (1 - self.beta_2 ** (self.iterations + 1))

        # Update parameters
        layer.weights += -self.current_learning_rate * weight_momentums_corrected / (np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases += -self.current_learning_rate * bias_momentums_corrected / (np.sqrt(bias_cache_corrected) + self.epsilon)
