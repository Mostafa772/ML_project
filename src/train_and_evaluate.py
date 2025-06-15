import random
from itertools import product

import numpy as np
import pandas as pd

from src.activation_functions import *
from src.loss_functions import *
from src.neural_network import Base_NN
from src.optimizers import *
from src.utils import *


class Train:
    def __init__(self, hyperparameters: dict[str], model: Base_NN):
        self.hp = hyperparameters
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []

    def train_and_evaluate(self, X_train, y_train, X_val, y_val):
        # Initialize components
        loss_function = MSE()
        optimizer = Optimizer_Adam(learning_rate=self.hp['learning_rate'], decay=selfy.hp['weight_decay'])

        self.train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []

        for epoch in range(n_epochs):
            batch_losses = []
            batch_accuracies = []

            for X_batch, y_batch in create_batches(X_train, y_train, batch_size):
                # Forward pass through model
                model.forward(X_batch, training=True)
                # Calculate loss through separate loss activation
                loss = loss_function.forward(model.output, y_batch)
                # Calculate accuracy
                predictions = np.round(model.output.squeeze())
                accuracy = np.mean(predictions == y_batch)

                # Backward pass
                loss_function.backward(model.output, y_batch)
                dvalues = loss_function.dinputs
                
                # Propagate gradients through model layers in reverse
                for layer in reversed(model.layers):
                    layer.backward(dvalues)
                    dvalues = layer.dinputs
                    
                    # Apply L1/L2 regularization to dense layers
                    if isinstance(layer, Layer_Dense):
                        if layer.l1 > 0:
                            layer.dweights += layer.l1 * np.sign(layer.weights)
                        if layer.l2 > 0:
                            layer.dweights += 2 * layer.l2 * layer.weights
                
                # Update parameters
                optimizer.pre_update_params()
                for layer in model.layers:
                    if isinstance(layer, Layer_Dense):
                        optimizer.update_params(layer)
                optimizer.post_update_params()
                
                batch_losses.append(loss)
                batch_accuracies.append(accuracy)

            # Epoch metrics
            epoch_loss = np.mean(batch_losses)
            epoch_accuracy = np.mean(batch_accuracies)
            train_losses.append(epoch_loss)
            train_accuracies.append(epoch_accuracy)
            
            # Validation
            X_val_input = X_val.values if isinstance(X_val, pd.DataFrame) else X_val
            y_val_input = y_val.values if isinstance(y_val, (pd.Series, pd.DataFrame)) else y_val

            model.forward(X_val_input, training=False)
            val_loss = loss_function.forward(model.output, y_val_input)
            val_predictions = np.round(model.output.squeeze())
            val_accuracy = np.mean(val_predictions == y_val.squeeze())

            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)

            # if epoch % 10 == 0:
            #     print(f"Epoch {epoch}: ", end="")
            #     print(f"Train Loss: {epoch_loss:.4f}, Acc: {epoch_accuracy*100:.2f}% | ", end="")
            #     print(f"Val Loss: {val_loss:.4f}, Acc: {val_accuracy*100:.2f}%")

        return model, val_accuracies[-1]