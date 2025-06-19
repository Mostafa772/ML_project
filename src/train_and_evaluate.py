import numpy as np
import random
import pandas as pd
from src.neural_network import *
from itertools import product
from src.optimizers import *
from src.activation_functions import *
from src.utils import *
from src.model_regularization import *
from src.layer import *
from src.random_search import *
from src.loss_functions import *
from src.data_preprocessing import *
from src.loss_functions import *
from src.ensemble.cascade_correlation import CascadeCorrelation


def train_and_evaluate(X_train, y_train, X_val, y_val, learning_rate, n_epochs, batch_size, weight_decay, patience, model):
        # Initialize components
        
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    loss_function = MSE()
    optimizer = Optimizer_Adam(learning_rate=learning_rate, decay=weight_decay)

    # Initialize early stopping
    early_stopping = EarlyStopping(patience, min_delta_loss=1e-5, min_delta_accuracy=0.001)

    # Before training loop:
    print("Data shapes:")
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    # print(f"Sample prediction: {model.forward(X_train[:1])}")
    # print(f"Initial loss: {loss_function.forward(model.output, y_train[:1])}")

    # Training loop
    for epoch in range(n_epochs):
        batch_losses = []
        batch_accuracies = []

        for X_batch, y_batch in create_batches(X_train, y_train, batch_size):
            # Forward pass
            model.forward(X_batch, training=True)

            # Loss and accuracy
            loss = loss_function.forward(model.output, y_batch)
            predictions = np.round(model.output.squeeze())
            accuracy = np.mean(predictions == y_batch.squeeze())

            # Backward pass
            loss_function.backward(model.output, y_batch)
            dvalues = loss_function.dinputs

            assert dvalues.shape == model.output.shape, \
                f"Gradient shape mismatch: {dvalues.shape} vs {model.output.shape}"
            
            i = 0
            for layer in reversed(model.layers):
                i=-1
                layer.backward(dvalues)
                dvalues = np.array(layer.dinputs)

                # Regularization
                if isinstance(layer, Layer_Dense):
                    if layer.l1 > 0:
                        layer.dweights += layer.l1 * np.sign(layer.weights)
                    if layer.l2 > 0:
                        layer.dweights += 2 * layer.l2 * layer.weights

            # Update weights
            optimizer.pre_update_params()
            for layer in model.layers:
                if isinstance(layer, Layer_Dense):
                    optimizer.update_params(layer)
            optimizer.post_update_params()

            batch_losses.append(loss)
            batch_accuracies.append(accuracy)

        # Epoch summary
        epoch_loss = np.mean(batch_losses)
        epoch_acc = np.mean(batch_accuracies)
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)

        # Validation
        X_val_input = X_val.values if isinstance(X_val, pd.DataFrame) else X_val
        y_val_input = y_val.values if isinstance(y_val, (pd.Series, pd.DataFrame)) else y_val

        model.forward(X_val_input, training=False)
        val_loss = loss_function.forward(model.output, y_val_input)
        val_predictions = np.round(model.output.squeeze())
        val_accuracy = np.mean(val_predictions == y_val.squeeze())

        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: ", end="")
            print(f"Train Loss: {epoch_loss:.4f}, Acc: {epoch_acc*100:.2f}% | ", end="")
            print(f"Val Loss: {val_loss:.4f}, Acc: {val_accuracy*100:.2f}%")

        # Early stopping check
        early_stopping.on_epoch_end(
            current_loss=val_loss,
            current_accuracy=val_accuracy,
            model=model,
            epoch=epoch
        )

        if early_stopping.stop_training:
            print(f"Early stopping at epoch {epoch}")
            # Restore best weights
            print(f"Restoring model weights from epoch {early_stopping.best_epoch}")
            early_stopping.restore_weights(model)
            # Cascade correlation
            if isinstance(model, CascadeCorrelation):
                if model.is_limit_reached():
                    break
                
                model.add_neuron()
                early_stopping.wait = 0
                early_stopping.patience -= int(early_stopping.patience / 10)
                early_stopping.stop_training = False
                print(f"Added new neuron at epoch {epoch} wiht val_loss {val_losses[-1]:.4f}")
                continue
            break

    return model, val_accuracies[-1]