import numpy as np

from activation_functions import Activation
from ensemble.cascade_correlation import CascadeCorrelation
from layer import Layer_Dense

class EarlyStopping:
    def __init__(self, patience=20, min_delta_loss=0.0, min_delta_accuracy=0.0):
        """
        Initialize the early stopping criteria.

        Parameters:
        - patience: Number of epochs to wait after the last improvement before stopping.
        - min_delta_loss: Minimum change in validation loss to qualify as an improvement.
        - min_delta_accuracy: Minimum change in validation accuracy to qualify as an improvement.
        - restore_best_weights: Whether to restore the model weights to the best epoch.
        """
        self.patience = patience
        self.min_delta_loss = min_delta_loss
        self.min_delta_accuracy = min_delta_accuracy
        self.best_weights = None
        self.best_epoch = 0
        self.best_loss = np.inf  # Track the best validation loss
        self.best_accuracy = -np.inf  # Track the best validation accuracy
        self.wait = 0  # Counter for epochs without improvement
        self.stop_training = False
    

    def on_epoch_end(self, current_loss: float, current_accuracy: float, model, epoch: int):
        """
        Call this at the end of each epoch to check if training should stop.

        Parameters:
        - current_loss: The validation loss from the current epoch.
        - current_accuracy: The validation accuracy from the current epoch.
        - model: The model being trained (to save the best weights).
        """
        # Check if loss has improved
        if current_loss < self.best_loss - self.min_delta_loss:
            self.save_weights(model)
            self.best_loss = current_loss
            self.best_accuracy = current_accuracy
            self.best_epoch = epoch
            self.wait = -1
        
        self.wait += 1
        if self.wait >= self.patience:
            self.stop_training = True
    
    def save_weights(self, model):
        self.best_weights = []
        for layer in model.layers:
            if isinstance(layer, Layer_Dense):
                weights = layer.weights.copy()
                biases = layer.biases.copy()
                self.best_weights.append((weights, biases))

    def restore_weights(self, model):
        if self.best_weights is None:
            print("Weights not restored")
            return

        if len(self.best_weights) != len(model):
            print("Weights cannot be restored, network size changed")
            return


        j = 0
        for layer in model.layers:
            if isinstance(layer, Layer_Dense):
                weights, biases = self.best_weights[j]
                layer.weights = weights.copy()
                layer.biases = biases.copy()
                j += 1