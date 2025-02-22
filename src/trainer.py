import numpy as np
import pandas as pd
from losses import *
from optimizers import *
from layer import *
from utils import *
from activation_functions import *
from model import *
from regularization.early_stopping import EarlyStopping



class NN_Set:
    def __init__(self, samples: np.ndarray, targets: np.ndarray):
        self.samples: np.ndarray = samples
        self.targets: np.ndarray = targets

class Trainer:
    def __init__(self, sets: dict[str, np.ndarray], loss: Loss = MSE()):
        self.training_set = NN_Set(sets["X_train"], sets["y_train"])
        self.validation_set = NN_Set(sets["X_val"], sets["y_val"])
        self.test_set = NN_Set(sets["X_test"], sets["y_test"])

        self.loss = loss
        self.epoch = 0
        self.model: NN | None = None
        self.optimizer: Optimizer_Base | None = None

    def run(self, hyperparams: dict) -> dict:
        print(hyperparams)
        
        self.model = NN(
            l1 = hyperparams["l1"],
            l2= hyperparams["l2"],
            input_size=6,
            hidden_sizes=[4, 6],
            output_size=1,
            hidden_activations=[hyperparams["activation"]],
            dropout_rates=[hyperparams["dropout_rate"]]
        )

        # early_stopping = EarlyStopping(
        #     patience=20,
        #     min_delta_loss=0.0001,
        #     min_delta_accuracy=0.0001
        # )
        
        self.optimizer = Optimizer_Adam(learning_rate=hyperparams["learning_rate"])
        # Before training loop:
        print("Data shapes:")
        print(f"X_train: {self.training_set.samples.shape}, y_train: {self.training_set.targets.shape}")
        output = self.model.forward(self.training_set.samples[:1])
        print(f"Sample prediction: {output}")  # Should output ~0.5
        print(f"Initial loss: {self.loss(output, self.training_set.targets[:1].squeeze())}")
        
        metrics = self.train(hyperparams["n_epochs"], hyperparams["batch_size"])
        
        test_accuracy, test_loss = self.test()
        
        plot_losses(metrics["train_losses"], metrics["val_losses"], label1="Training Loss", label2="Validation Loss", title="Loss Over Epochs")
        plot_accuracies(metrics["train_accuracies"], metrics["val_accuracies"], label1="Training Accuracies", label2="Validation Accuracies", title="Accuracy Over Epochs")

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
        batch_losses = []
        batch_accuracies = []
        for _ in range(epochs):
            losses, accuracies = self.train_epoch(batch_size)
            batch_losses.append(losses)
            batch_accuracies.append(accuracies)

            # Epoch metrics
            epoch_loss = np.mean(batch_losses)
            epoch_acc = np.mean(batch_accuracies)
            train_losses.append(epoch_loss)
            train_accuracies.append(epoch_acc)

            val_loss, val_accuracy = self.validate()
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)

        return {
            "train_losses" : train_losses,
            "train_accuracies" : train_accuracies,
            "val_losses" : val_losses,
            "val_accuracies" : val_accuracies
        }

    def train_epoch(self, batch_size: int) -> tuple:
        assert self.model is not None
        assert self.optimizer is not None

        batch_losses = []
        batch_accuracies = []
        
        for X_batch, y_batch in create_batches(self.training_set.samples, self.training_set.targets, batch_size):
            # Forward pass
            output = self.model.forward(X_batch, training=True)
            
            # Calculate loss
            loss = self.loss(output, y_batch)

            predictions = np.round(output.squeeze())
            accuracy = np.mean(predictions == y_batch.squeeze())

            # Backward pass with shape validation
            dvalues = self.loss.backward(output, y_batch)
            
            # Verify gradient shape matches output
            assert dvalues.shape == output.shape, \
                f"Gradient shape mismatch: {dvalues.shape} vs {output.shape}"
            
            # Propagate gradients
            # self.optimizer.pre_update_params()
            for layer in reversed(self.model.layers):
                # Ensure numpy arrays
                dvalues = to_ndarray(dvalues)
                dvalues = layer.backward(dvalues)
                self.optimizer.update_params(layer)
                
            # self.optimizer.post_update_params()
            
            batch_losses.append(loss)
            batch_accuracies.append(accuracy)
        
        return batch_losses, batch_accuracies

    def test(self) -> tuple:
        assert self.model is not None

        output = self.model.forward(self.test_set.samples, training=False)
        
        predictions = np.round(output.squeeze())
        loss = self.loss(predictions, self.test_set.targets)
        
        # Calculate accuracy for the test set
        y_true = self.test_set.targets
        if len(self.test_set.targets.shape) == 2:
            y_true = np.argmax(self.test_set.targets, axis=1)

        accuracy = np.mean(predictions == y_true)

        return accuracy, loss

    def validate(self) -> tuple:
        assert self.model is not None

        samples = self.validation_set.samples
        if isinstance(self.validation_set.samples, pd.DataFrame):
            samples = self.validation_set.samples.values

        targets = self.validation_set.targets
        if isinstance(self.validation_set.targets, (pd.Series, pd.DataFrame)):
            targets = self.validation_set.targets.values

        output = self.model.forward(samples, training=False)
        val_predictions = np.round(output.squeeze())
        
        val_loss = self.loss(output, targets)
        val_accuracy = np.mean(val_predictions == self.validation_set.targets.squeeze())
        return val_loss, val_accuracy


        
