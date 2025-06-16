import numpy as np
import random
import pandas as pd
from src.neural_network import *
from itertools import product
from src.optimizers import *
from src.activation_functions import *
from src.utils import *
from src.layer import *
from src.random_search import *
from src.loss_functions import *
from src.data_preprocessing import *
from src.loss_functions import *


def train_and_evaluate(X_train, y_train, X_val, y_val, learning_rate, n_epochs, batch_size, weight_decay, model):
    # Initialize components
    loss_function = MSE()
    optimizer = Optimizer_Adam(learning_rate=learning_rate, decay=weight_decay)

    train_losses = []
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

# def train_and_evaluate(X_train, y_train, X_val, y_val,
#                        learning_rate, n_epochs, batch_size, weight_decay,
#                        model, l1=0.0, l2=0.0, dropout_rate=0.0,
#                        hidden_activation=None, use_batch_norm=False,
#                        early_stopping=None, verbose=False):
#     """
#     Train the model and return the trained model along with validation accuracy.
#     Supports dropout, batch norm, L1/L2 regularization, early stopping, and custom activation.
#     """

#     loss_function = MSE()
#     optimizer = Optimizer_Adam(learning_rate=learning_rate, decay=weight_decay)

#     train_losses = []
#     train_accuracies = []
#     val_losses = []
#     val_accuracies = []

#     for epoch in range(n_epochs):
#         batch_losses = []
#         batch_accuracies = []

#         for X_batch, y_batch in create_batches(X_train, y_train, batch_size):
#             model.forward(X_batch, training=True)

#             loss = loss_function.forward(model.output, y_batch)
#             predictions = np.round(model.output.squeeze())
#             accuracy = np.mean(predictions == y_batch)

#             loss_function.backward(model.output, y_batch)
#             dvalues = loss_function.dinputs

#             # Backpropagation through model
#             for layer in reversed(model.layers):
#                 layer.backward(dvalues)
#                 dvalues = layer.dinputs

#                 if isinstance(layer, Layer_Dense):
#                     if l1 > 0:
#                         layer.dweights += l1 * np.sign(layer.weights)
#                     if l2 > 0:
#                         layer.dweights += 2 * l2 * layer.weights

#             # Weight update
#             optimizer.pre_update_params()
#             for layer in model.layers:
#                 if isinstance(layer, Layer_Dense):
#                     optimizer.update_params(layer)
#             optimizer.post_update_params()

#             batch_losses.append(loss)
#             batch_accuracies.append(accuracy)

#         # Epoch statistics
#         epoch_loss = np.mean(batch_losses)
#         epoch_accuracy = np.mean(batch_accuracies)
#         train_losses.append(epoch_loss)
#         train_accuracies.append(epoch_accuracy)

#         # Validation step
#         X_val_input = X_val.values if isinstance(
#             X_val, pd.DataFrame) else X_val
#         y_val_input = y_val.values if isinstance(
#             y_val, (pd.Series, pd.DataFrame)) else y_val

#         model.forward(X_val_input, training=False)
#         val_loss = loss_function.forward(model.output, y_val_input)
#         val_predictions = np.round(model.output.squeeze())
#         val_accuracy = np.mean(val_predictions == y_val.squeeze())

#         val_losses.append(val_loss)
#         val_accuracies.append(val_accuracy)

#         # Optional: Logging
#         if verbose and epoch % 10 == 0:
#             print(f"Epoch {epoch}: Train Loss = {epoch_loss:.4f}, Acc = {epoch_accuracy*100:.2f}%, "
#                   f"Val Loss = {val_loss:.4f}, Acc = {val_accuracy*100:.2f}%")

#         # Optional: Early stopping
#         if early_stopping:
#             early_stopping.on_epoch_end(val_loss, val_accuracy, model, epoch)
#             if early_stopping.stop_training:
#                 if verbose:
#                     print(f"Early stopping at epoch {epoch}")
#                 break

#     # Optional: Restore best weights
#     if early_stopping and early_stopping.best_weights:
#         early_stopping.restore_weights(model)

#     return model, val_accuracies[-1] if val_accuracies else 0.0
