import numpy as np


class EarlyStopping:
    def __init__(self, patience=10, min_delta_loss=0.0, min_delta_accuracy=0.0, restore_best_weights=True):
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
        self.restore_best_weights = restore_best_weights
        self.best_weights = None
        self.best_epoch = 0
        self.best_loss = np.inf  # Track the best validation loss
        self.best_accuracy = -np.inf  # Track the best validation accuracy
        self.wait = 0  # Counter for epochs without improvement
        self.stop_training = False

    def on_epoch_end(self, current_loss, current_accuracy, model, epoch):
        """
        Call this at the end of each epoch to check if training should stop.

        Parameters:
        - current_loss: The validation loss from the current epoch.
        - current_accuracy: The validation accuracy from the current epoch.
        - model: The model being trained (to save the best weights).
        """
        # Check if either loss or accuracy has improved
        loss_improved = current_loss < self.best_loss - self.min_delta_loss
        accuracy_improved = current_accuracy > self.best_accuracy + self.min_delta_accuracy

        if loss_improved or accuracy_improved:
            # Improvement detected
            if loss_improved:
                self.best_loss = current_loss
            if accuracy_improved:
                self.best_accuracy = current_accuracy
            self.best_epoch = epoch
            self.wait = 0
            # if self.restore_best_weights:
            #     # Save the best weights
            #     self.best_weights = [layer.weights.copy() for layer in model.layers]
        else:
            # No improvement
            self.wait += 1
            if self.wait >= self.patience:
                self.stop_training = True

    def restore_weights(self, model):
        """
        Restore the model's weights to the best epoch.

        Parameters:
        - model: The model to restore the weights to.
        """
        if self.best_weights is not None:
            for layer, weights in zip(model.layers, self.best_weights):
                layer.weights = weights
                
                
                
class Dropout:
    def __init__(self, rate):
        """
        Initialize a dropout layer.
        
        Parameters:
        - rate: Dropout rate (fraction of inputs to drop)
        """
        self.rate = 1 - rate  # Store keep rate instead of drop rate
        self.mask = None
        
    def forward(self, inputs, training=True):
        """
        Perform the forward pass with dropout.
        
        Parameters:
        - inputs: Input data
        - training: Boolean indicating training mode
        """
        self.inputs = inputs
        
        if training:
            self.mask = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate
            self.output = inputs * self.mask
        else:
            self.output = inputs
            
    def backward(self, dvalues):
        """
        Perform the backward pass through dropout.
        
        Parameters:
        - dvalues: Gradient of the loss with respect to the output
        """
        self.dinputs = dvalues * self.mask