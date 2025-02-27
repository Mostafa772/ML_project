from abc import ABC, abstractmethod
import numpy as np
from src.activation_functions import Activation_Softmax
from src.regularization.regularization import Regularization

class Loss(ABC):
    def __init__(self, regularizations: list[Regularization]):
        self.regularizations = regularizations

    @abstractmethod
    def forward(self, y_pred, y_true):
        raise NotImplemented

    @abstractmethod
    def backward(self, dvalues, y_true) -> np.ndarray:
        raise NotImplemented

    def __call__(self, y_pred, y_true, weights: np.ndarray | None = None) -> np.ndarray:
        """
        __call__ implements the loss calculation but adds the regularizations mechanisms
        """
        loss = self.forward(y_pred, y_true)
        for regularization in self.regularizations:
            loss += regularization(weights)
        return loss

class Loss_CategoricalCrossentropy(Loss):

    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray):
        y_pred = np.clip(y_pred, 1e-12, 1 - 1e-12)  # Avoid log(0)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def backward(self, dvalues, y_true) -> np.ndarray:
        return dvalues - y_true


class Activation_Softmax_Loss_CategoricalCrossentropy():

    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()


    def forward(self, inputs, y_true):
        # Output layer's activation function
        self.activation.forward(inputs)

        # Set the output
        self.output = self.activation.output

        return self.loss.calculate(self.output, y_true)

    def backward(self, dvalues, y_true):
        samples = len(dvalues)

        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        self.dinputs = dvalues.copy()

        self.dinputs[range(samples), y_true] -= 1

        self.dinputs = self.dinputs / samples

class MSE(Loss):
    def __init__(self):
        self.dinputs = None
        self.output = None
        
    def forward(self, y_pred, y_true):
        # Remove the shape condition - always calculate loss
        self.output = np.mean((y_pred - y_true)**2)
        return self.output
    
    def backward(self, dvalues, y_true) -> np.ndarray:
        samples = len(dvalues)
        self.dinputs = 2 * (dvalues - y_true) / samples
        return self.dinputs
        