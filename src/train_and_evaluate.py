import numpy as np
import pandas as pd

from scheduler import LearningRateScheduler
from src.activation_functions import *
from src.early_stopping import EarlyStopping
from src.layer import Layer_Dense
from src.loss_functions import *
from src.neural_network import Base_NN
from src.cascade_correlation import CascadeCorrelation
from src.optimizers import *
from src.utils import *


class Train:
    def __init__(self, hyperparameters: dict[str, float | int], model: Base_NN, regression: bool = False, verbose: bool = False):
        self.hp = hyperparameters
        self.loss_function = MSE()
        self.train_losses = np.array([])
        self.train_scores = np.array([])
        self.val_losses = np.array([])
        self.val_scores = np.array([])
        self.test_loss = None
        self.test_score = None
        self.model = model
        self.regression=regression
        self.verbose = verbose

    def train_and_evaluate(self, X_train, y_train, X_val, y_val):
        self.train_losses = []
        self.train_scores = []
        self.val_losses = []
        self.val_scores = []
        
        X_val = X_val.values if isinstance(X_val, pd.DataFrame) else X_val
        y_val = y_val.values if isinstance(y_val, (pd.Series, pd.DataFrame)) else y_val

        optimizer = Optimizer_Adam(learning_rate=self.hp['learning_rate'], decay=self.hp['weight_decay'])
        sched = LearningRateScheduler(optimizer, self.hp['sched_decay'], window=int(self.hp['patience']/2))

        # Initialize early stopping
        assert isinstance(self.hp['patience'], int)
        early_stopping = EarlyStopping(patience=self.hp['patience'], min_delta_loss=1e-5, min_delta_accuracy=0.001)

        # Before training loop:
        if self.verbose:
            print("Data shapes:")
            print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
            print(f"Hyperparams: {self.hp}")

        # Training loop
        assert isinstance(self.hp['n_epochs'], int)
        for epoch in range(self.hp['n_epochs']):
            batch_losses = []
            batch_scores = []

            for X_batch, y_batch in create_batches(X_train, y_train, self.hp['batch_size']):
                # Forward pass
                self.model.forward(X_batch, training=True)

                # Loss and score
                loss = self.loss_function.forward(self.model.output, y_batch)
                predictions = np.round(self.model.output.squeeze())
                if not self.regression:
                    predictions = np.round(self.model.output.squeeze())
                    score = np.mean(predictions == y_batch.squeeze())
                else:
                    score = r2_score_global(y_batch, self.model.output)

                # Backward pass
                self.loss_function.backward(self.model.output, y_batch)
                dvalues = self.loss_function.dinputs

                assert dvalues.shape == self.model.output.shape, \
                    f"Gradient shape mismatch: {dvalues.shape} vs {self.model.output.shape}"
                
                
                for layer in reversed(self.model.layers):
                    layer.backward(dvalues)
                    dvalues = np.array(layer.dinputs)

                # Update weights
                optimizer.pre_update_params()
                for layer in self.model.layers:
                    if isinstance(layer, Layer_Dense):
                        optimizer.update_params(layer)
                optimizer.post_update_params()


                batch_losses.append(loss)
                batch_scores.append(score)

            # Epoch summary
            epoch_loss = np.mean(batch_losses)
            epoch_acc = np.mean(batch_scores)
            self.train_losses.append(epoch_loss)
            self.train_scores.append(epoch_acc)

            
            # Validation
            self.model.forward(X_val, training=False)
            val_loss = self.loss_function.forward(self.model.output, y_val)
            val_predictions = np.round(self.model.output.squeeze())
            if not self.regression:
                val_predictions = np.round(self.model.output.squeeze())
                val_score = np.mean(val_predictions == y_val.squeeze())
            else:
                val_score = r2_score_global(y_val,self.model.output)

            self.val_losses.append(val_loss)
            self.val_scores.append(val_score)

            if epoch % 10 == 0 and self.verbose:
                print(f"Epoch {epoch}: ", end="")
                print(f"Train Loss: {epoch_loss:.4f}, Acc: {epoch_acc*100:.2f}% | ", end="")
                print(f"Val Loss: {val_loss:.4f}, Acc: {val_score*100:.2f}%")
            
            # Learning rate scheduler
            sched.at_epoch_end(val_loss)

            # Early stopping check
            early_stopping.on_epoch_end(
                current_loss=val_loss,
                current_accuracy=val_score,
                model=self.model,
                epoch=epoch
            )

            if early_stopping.stop_training:
                if self.verbose:
                    print(f"Early stopping at epoch {epoch}")
                    print(f"Restoring model weights from epoch {early_stopping.best_epoch}")
                # Restore best weights
                early_stopping.restore_weights(self.model)
                # Cascade correlation
                if isinstance(self.model, CascadeCorrelation):
                    if self.model.is_limit_reached():
                        break
                    
                    self.model.add_neuron()
                    early_stopping.wait = 0
                    early_stopping.patience -= int(early_stopping.patience / 10)
                    early_stopping.stop_training = False
                    if self.verbose:
                        print(f"Added new neuron at epoch {epoch} with val_loss {self.val_losses[-1]:.4f}")
                    continue
                break

        if self.verbose: 
            print(f"Final Validation score: {self.val_scores[-1]:.4f}")
        return self.model, self.val_scores[-1]
    
    def test(self, X_test, y_test) -> tuple[float, float]:
        self.model.forward(X_test, training=False)
        self.test_loss = self.loss_function.forward(self.model.output.squeeze(), y_test)

        if not self.regression:
            predictions = np.round(self.model.output.squeeze())
            y_true = np.argmax(y_test, axis=1) if y_test.ndim > 1 else y_test
            self.test_score = np.mean(predictions == y_true.squeeze())
        else:
            self.test_score = r2_score_global(y_test,self.model.output)

        if self.verbose:
            print(f"Test score: {self.test_score:.4f}")
        return self.test_loss, self.test_score

    def plot(self, score=False):
         # Plot training progress
        plot_losses(self.train_losses, self.val_losses, self.test_loss,
                    label1="Training Loss", label2="Validation Loss",
                    title="Loss Over Epochs")

        if score:
            plot_scores(self.train_scores, self.val_scores, self.test_score,
                        label1="Training scores", label2="Validation scores",
                        title="score Over Epochs")
    