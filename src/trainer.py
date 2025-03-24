import numpy as np
import pandas as pd

from activation_functions import *
from layer import *
from losses import *
from model import *
from optimizers import *
from regularization import early_stopping
from regularization.early_stopping import EarlyStopping
from utils import *


class NN_Set:
    def __init__(self, samples: np.ndarray, targets: np.ndarray):
        self.samples: np.ndarray = to_ndarray(samples)
        self.targets: np.ndarray = to_ndarray(targets)

class Trainer:
    def __init__(self, hyperparams: dict, sets: dict[str, np.ndarray], loss: Loss = MSE()):
        self.training_set = NN_Set(sets["X_train"], sets["y_train"])
        self.validation_set = NN_Set(sets["X_val"], sets["y_val"])
        self.test_set = NN_Set(sets["X_test"], sets["y_test"])

        self.loss: Loss = loss
        self.epoch = 0
        self.max_epochs: int = hyperparams["n_epochs"]
        self.batch_size: int = hyperparams["batch_size"]
        self.optimizer = Optimizer_Adam(learning_rate=hyperparams["learning_rate"])
        self.model = NN(
            l1 = hyperparams["l1"],
            l2= hyperparams["l2"],
            input_size=6,
            hidden_sizes= hyperparams["hidden_sizes"],
            output_size= hyperparams["output_size"],
            hidden_activations=[hyperparams["activation"]],
            dropout_rates=[hyperparams["dropout_rate"]]
        )
        self.early_stopping = EarlyStopping(patience=hyperparams["patience"])

    def run(self) -> dict:        
        # Before training loop:
        print(f"Data shapes: \nX_train: {self.training_set.samples.shape}, y_train: {self.training_set.targets.shape}")        
        metrics = self.train(self.max_epochs, self.batch_size)
        
        test_accuracy, test_loss = self.test()
        
        plot_losses(metrics["train_losses"], metrics["val_losses"], test_loss)
        plot_accuracies(metrics["train_accuracies"], metrics["val_accuracies"], test_accuracy)

        return {
            "train_loss" : metrics["train_losses"],
            "validation_loss" : metrics["val_losses"],
            "test_loss" : test_loss,
            "train_accuracy" : metrics["train_accuracies"],
            "validation_accuracy" : metrics["val_accuracies"],
            "test_accuracy" : test_accuracy
        }

    def train(self, epochs:int, batch_size: int = 1) -> dict:
        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []

        for _ in range(epochs):
            loss, accuracy = self.train_epoch(batch_size)

            # Epoch metrics
            train_losses.append(loss)
            train_accuracies.append(accuracy)

            val_loss, val_accuracy = self.validate()
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)

            self.early_stopping.on_epoch_end(val_loss, val_accuracy, self.model)
            if self.early_stopping.stop_training:
                break
            
        print(f"Training stopped at epoch {self.early_stopping.best_epoch + self.early_stopping.wait} with validation accuracy: {self.early_stopping.best_accuracy} and loss {self.early_stopping.best_loss}")
        self.early_stopping.restore_weights(self.model)

        return {
            "train_losses" : train_losses,
            "train_accuracies" : train_accuracies,
            "val_losses" : val_losses,
            "val_accuracies" : val_accuracies
        }

    def train_epoch(self, batch_size: int) -> tuple:
        batch_losses = []
        batch_accuracies = []

        for X_batch, y_batch in create_batches(self.training_set.samples, self.training_set.targets, batch_size):
            # print(f"batch size {X_batch.shape}") Later to fix
            # Forward pass
            output = self.model.forward(X_batch, training=True)
            
            # Calculate loss
            loss = self.loss(output, y_batch)

            predictions = np.round(output.squeeze())
            accuracy = np.mean(predictions == y_batch.squeeze())

            # Backward pass with shape validation
            dvalues = self.loss.backward(output, y_batch)
            
            # Verify gradient shape matches output
            assert dvalues.shape == output.shape, f"Gradient shape mismatch: {dvalues.shape} vs {output.shape}"
            
            # Propagate gradients
            self.optimizer.pre_update_params()
            for layer in reversed(self.model.layers):
                # Ensure numpy arrays
                dvalues = to_ndarray(dvalues)
                dvalues = layer.backward(dvalues)
                self.optimizer.update_params(layer)
                
            self.optimizer.post_update_params()            
            
            batch_losses.append(loss)
            batch_accuracies.append(accuracy)

        return np.mean(batch_losses), np.mean(batch_accuracies)

    def test(self) -> tuple:
        output = self.model.forward(self.test_set.samples, training=False)
        
        predictions = np.round(output.squeeze())
        loss = self.loss.forward(predictions, self.test_set.targets)
        
        # Calculate accuracy for the test set
        y_true = self.test_set.targets
        if len(self.test_set.targets.shape) == 2:
            y_true = np.argmax(self.test_set.targets, axis=1)

        accuracy = np.mean(predictions == y_true)

        return accuracy, loss

    def validate(self) -> tuple:
        samples = self.validation_set.samples
        targets = self.validation_set.targets

        output = self.model.forward(samples, training=False)
        val_predictions = np.round(output.squeeze())
        
        val_loss = self.loss.forward(output, targets)
        val_accuracy = np.mean(val_predictions == self.validation_set.targets.squeeze())
        return val_loss, val_accuracy


        
