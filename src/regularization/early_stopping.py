import numpy as np

from model import NN


class EarlyStopping:
    def __init__(self, patience, min_delta_loss :float = 1e-14, min_delta_accuracy :float = 1e-14):
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
        self.wait = -1  # Counter for epochs without improvement
        self.stop_training = False

    def on_epoch_end(self, current_loss: float, current_accuracy: float, model: NN) -> bool:
        """
        Call this at the end of each epoch.
        Returns a flag indicating wether or not training should stop.

        Parameters:
        - current_loss: The validation loss from the current epoch.
        - current_accuracy: The validation accuracy from the current epoch.
        - model: The model being trained (to save the best weights).
        """
        # Check if either loss or accuracy has improved
        loss_improved = current_loss < self.best_loss + self.min_delta_loss
        accuracy_improved = current_accuracy > self.best_accuracy + self.min_delta_accuracy

        if loss_improved:
            self.best_loss = current_loss
        if accuracy_improved:
            self.best_accuracy = current_accuracy

        if loss_improved or accuracy_improved:
            # Improvement detected
            self.best_weights = [layer.weights for layer in model.layers]
            self.best_epoch += self.wait + 1
            self.wait = 0
        else:
            # No improvement
            self.wait += 1
            
        if self.wait >= self.patience:
            self.stop_training = True

        return self.stop_training

    def restore_weights(self, model: NN):
        """
        Restore the model's weights to the best epoch.

        Parameters:
        - model: The model to restore the weights to.
        """
        assert self.best_weights is not None, f"Called restore_weights of Early Stopping when no weights were saved. \nDid you forget to call on_epoch_end?"
        
        for layer, weights in zip(model.layers, self.best_weights):
            layer.weights = weights
