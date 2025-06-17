import random
from itertools import product

import numpy as np
import pandas as pd

from src.activation_functions import *
from src.early_stopping import EarlyStopping
from src.layer import Layer_Dense
from src.loss_functions import *
from src.data_preprocessing import *
from src.loss_functions import *
from src.early_stopping import EarlyStopping
from src.ensemble.cascade_correlation import CascadeCorrelation


def train_and_evaluate(X_train, y_train, X_val, y_val, learning_rate, n_epochs, batch_size,
                       weight_decay, patience, model):
    # Initialize components
    loss_function = MSE()
    optimizer = Optimizer_Adam(learning_rate=learning_rate, decay=weight_decay)
    early_stopping = EarlyStopping(patience=patience, min_delta_loss=1e-5, min_delta_accuracy=0.001)
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    for epoch in range(n_epochs):
        batch_losses = []
        batch_accuracies = []

            for X_batch, y_batch in create_batches(X_train, y_train, self.hp['batch_size']):
                # Forward pass
                self.model.forward(X_batch, training=True)

                # Loss and accuracy
                loss = self.loss_function.forward(self.model.output, y_batch)
                predictions = np.round(self.model.output.squeeze())
                accuracy = np.mean(predictions == y_batch.squeeze())

            # Backward pass
            loss_function.backward(model.output, y_batch)
            dvalues = loss_function.dinputs
            
            # Propagate gradients through model layers in reverse
            for layer in reversed(model.layers):
                layer.backward(dvalues)
                dvalues = layer.dinputs
            
            # Update parameters
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
            self.train_losses.append(epoch_loss)
            self.train_accuracies.append(epoch_acc)

            
            # Validation
            self.model.forward(X_val, training=False)
            val_loss = self.loss_function.forward(self.model.output, y_val)
            val_predictions = np.round(self.model.output.squeeze())
            val_accuracy = np.mean(val_predictions == y_val.squeeze())

        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
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
        # if epoch % 10 == 0:
        #     print(f"Epoch {epoch}: ", end="")
        #     print(f"Train Loss: {epoch_loss:.4f}, Acc: {epoch_accuracy*100:.2f}% | ", end="")
        #     print(f"Val Loss: {val_loss:.4f}, Acc: {val_accuracy*100:.2f}%")

            # Early stopping check
            early_stopping.on_epoch_end(
                current_loss=val_loss,
                current_accuracy=val_accuracy,
                model=self.model,
                epoch=epoch
            )

            if early_stopping.stop_training:
                print(f"Early stopping at epoch {epoch}")
                # Restore best weights
                print(f"Restoring model weights from epoch {early_stopping.best_epoch}")
                early_stopping.restore_weights(self.model)
                # Cascade correlation
                if isinstance(self.model, CascadeCorrelation):
                    if self.model.is_limit_reached():
                        break
                    
                    self.model.add_neuron()
                    early_stopping.wait = 0
                    early_stopping.patience -= int(early_stopping.patience / 10)
                    early_stopping.stop_training = False
                    print(f"Added new neuron at epoch {epoch} wiht val_loss {self.val_losses[-1]:.4f}")
                    continue
                break

        print(f"Final Validation Accuracy: {self.val_accuracies[-1]:.4f}")
        return self.model, self.val_accuracies[-1]
    
    def test(self, X_test, y_test):
        self.model.forward(X_test, training=False)
        self.test_loss = self.loss_function.forward(self.model.output.squeeze(), y_test)

        predictions = np.round(self.model.output.squeeze())
        y_true = np.argmax(y_test, axis=1) if y_test.ndim > 1 else y_test
        self.test_accuracy = np.mean(predictions == y_true)
        print(f"Test Accuracy: {self.test_accuracy:.4f}")

    def plot(self, accuracy=False):
         # Plot training progress
        plot_losses(self.train_losses, self.val_losses, self.test_loss,
                    label1="Training Loss", label2="Validation Loss",
                    title="Loss Over Epochs")

        if accuracy:
            plot_accuracies(self.train_accuracies, self.val_accuracies, self.test_accuracy,
                        label1="Training Accuracies", label2="Validation Accuracies",
                        title="Accuracy Over Epochs")
    
    