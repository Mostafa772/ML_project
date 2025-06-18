import numpy as np

from ensemble.cascade_correlation import CascadeCorrelation

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
            if not hasattr(layer, 'weights'):
                continue
            self.best_weights.append(layer.weights)

    def restore_weights(self, model):
        """
        Restore the model's weights to the best epoch.

        Parameters:
        - model: The model to restore the weights to.
        """
        if self.best_weights is None:
            print("Weights not restored")
            return

        if isinstance(model, CascadeCorrelation) and len(self.best_weights) != len(model.layers)/2:
            print("Weights cannot be restored, network size changed")
            return

        for layer, weights in zip(model.layers, self.best_weights):
            if not hasattr(layer, 'weights'):
                continue
            layer.weights = weights
                
                
