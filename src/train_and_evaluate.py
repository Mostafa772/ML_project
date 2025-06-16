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
from src.early_stopping import EarlyStopping
from src.ensemble.cascade_correlation import CascadeCorrelation


def train_and_evaluate(X_train, y_train, X_val, y_val, learning_rate, n_epochs, batch_size,
                       weight_decay, patience, model, regression=False):
    # Determine task type
    print("regression: ", regression)

    # Initialize components
    loss_function = MSE()
    optimizer = Optimizer_Adam(learning_rate=learning_rate, decay=weight_decay)
    early_stopping = EarlyStopping(patience=patience, min_delta_loss=1e-5, min_delta_accuracy=0.001)

    train_losses = []
    train_scores = []
    val_losses = []
    val_scores = []

    for epoch in range(n_epochs):
        batch_losses = []
        batch_scores = []

        for X_batch, y_batch in create_batches(X_train, y_train, batch_size):
            model.forward(X_batch, training=True)
            loss = loss_function.forward(model.output, y_batch)

            if not regression:
                predictions = np.round(model.output.squeeze())
                score = np.mean(predictions == y_batch.squeeze())
            else:
                score = r2_score_global(y_batch, model.output)

            # Backward pass
            loss_function.backward(model.output, y_batch)
            dvalues = loss_function.dinputs
            for layer in reversed(model.layers):
                layer.backward(dvalues)
                dvalues = layer.dinputs

            # Update
            optimizer.pre_update_params()
            for layer in model.layers:
                if isinstance(layer, Layer_Dense):
                    optimizer.update_params(layer)
            optimizer.post_update_params()

            batch_losses.append(loss)
            batch_scores.append(score)

        epoch_loss = np.mean(batch_losses)
        epoch_score = np.mean(batch_scores)
        train_losses.append(epoch_loss)
        train_scores.append(epoch_score)

        # Validation
        X_val_input = X_val.values if isinstance(X_val, pd.DataFrame) else X_val
        y_val_input = y_val.values if isinstance(y_val, (pd.Series, pd.DataFrame)) else y_val

        model.forward(X_val_input, training=False)
        val_loss = loss_function.forward(model.output, y_val_input)

        if not regression:
            val_predictions = np.round(model.output.squeeze())
            val_score = np.mean(val_predictions == y_val_input.squeeze())
        else:
            val_score = r2_score_global(y_val_input, model.output)

        val_losses.append(val_loss)
        val_scores.append(val_score)

        early_stopping.on_epoch_end(
            current_loss=val_loss,
            current_accuracy=val_score,
            model=model,
            epoch=epoch
        )

        if early_stopping.stop_training:
            print(f"Early stopping at epoch {epoch}")
            print(f"Restoring model weights from epoch {early_stopping.best_epoch}")
            early_stopping.restore_weights(model)
            if isinstance(model, CascadeCorrelation):
                if model.is_limit_reached():
                    break
                model.add_neuron()
                early_stopping.wait = 0
                early_stopping.patience -= int(early_stopping.patience / 10)
                early_stopping.stop_training = False
                print(f"Added new neuron at epoch {epoch} with val_loss {val_losses[-1]:.4f}")
                continue
            break

    return model, val_scores[-1]