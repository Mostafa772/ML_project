import numpy as np
from losses import *
from optimizers import *
from layer import *
from utils import *
from activation_functions import *
from model import *



class NN_Set:
    def __init__(self, samples: np.ndarray, targets: np.ndarray):
        self.samples: np.ndarray = samples
        self.targets: np.ndarray = targets

class Trainer:
    def __init__(self, sets: dict[str, np.ndarray]):
        self.train = NN_Set(sets["X_train"], sets["y_train"])
        self.validate = NN_Set(sets["X_val"], sets["y_val"])
        self.test = NN_Set(sets["X_test"], sets["y_test"])

        self.loss = MSE()
        self.epoch = 0
        self.model: NN | None = None
        self.optimizer: Optimizer_Base | None = None

    def run(self, hyperparams: dict):
        print(hyperparams)
        for i in hyperparams: 
            learning_rate, l1, l2, dropout_rate, batch_size, n_epochs, activation = i.values()
            self.model = NN(
                l1=l1,
                l2=l2,
                input_size=6,
                hidden_sizes=[1],
                output_size=1,
                hidden_activations=[activation],
                dropout_rates=[dropout_rate]
            )
            train_losses = []
            train_accuracies = []
            val_losses = []
            val_accuracies = []
            batch_losses = []
            batch_accuracies = []

            # early_stopping = EarlyStopping(
            #     patience=20,
            #     min_delta_loss=0.0001,
            #     min_delta_accuracy=0.0001,
            #     restore_best_weights=True
            # )
            # loss_activation = MSE()

            self.optimizer = Optimizer_Adam(learning_rate=learning_rate)
            # Before training loop:
            print("Data shapes:")
            print(f"X_train: {self.train.samples.shape}, y_train: {self.train.targets.shape}")
            output = self.model.forward(self.train.samples[:1])
            print(f"Sample prediction: {output}")  # Should output ~0.5
            print(f"Initial loss: {self.loss.forward(output, self.train.targets[:1].squeeze())}")
            # Training loop
            for _ in range(n_epochs):
                losses, accuracies = self.train_epoch(batch_size)
                batch_losses.append(losses)
                batch_accuracies.append(accuracies)

                # Epoch metrics
                epoch_loss = np.mean(batch_losses)
                epoch_acc = np.mean(batch_accuracies)
                train_losses.append(epoch_loss)
                train_accuracies.append(epoch_acc)
                print(epoch_loss, epoch_acc)
                # Validation pass
                self.model.forward(self.validate.samples.values if isinstance(self.validate.samples, pd.DataFrame) else self.validate.samples, 
                                training=False)
                # print(model.output)
                val_loss = self.loss.forward(self.model.output, self.validate.targets.values if isinstance(self.validate.targets, (pd.Series, pd.DataFrame)) else self.validate.targets)
                val_predictions = np.round(self.model.output.squeeze())
                val_accuracy = np.mean(val_predictions == self.validate.targets.squeeze())
                
                val_losses.append(val_loss)
                val_accuracies.append(val_accuracy)
            # print(len(val_accuracies), len(train_accuracies))
            
            
            self.model.forward(self.test.samples, training=False)
            # Compute softmax probabilities for the test output
            # print(X_test.shape, y_test.shape)
            # print(model.output, y_test)
            self.loss.forward(self.model.output.squeeze(), self.test.targets) 
            # Calculate accuracy for the test set
            predictions = np.round(self.model.output.squeeze())
            if len(self.test.targets.shape) == 2:
                y_true = np.argmax(self.test.targets, axis=1) 
            else:
                y_true = self.test.targets

            # Compute test accuracy
            test_accuracy = np.mean(predictions == y_true)
            # test_accuracies.append(test_accuracy)
            print(f"Test Accuracy: {test_accuracy:.4f}")
            plot_losses(train_losses, val_losses, label1="Training Loss", label2="Validation Loss", title="Loss Over Epochs")
            plot_accuracies(train_accuracies, val_accuracies, label1="Training Accuracies", label2="Validation Accuracies", title="Accuracy Over Epochs")

            return train_losses, val_losses

    def train_epoch(self, batch_size) -> tuple:
        assert self.model is not None

        batch_losses = []
        batch_accuracies = []
        
        for X_batch, y_batch in create_batches(self.train.samples, self.train.targets, batch_size):
            # Forward pass
            output = self.model.forward(X_batch, training=True)
            
            # Calculate loss
            loss = self.loss.forward(output, y_batch)
            
            predictions = np.round(output.squeeze())
            accuracy = np.mean(predictions == y_batch.squeeze())

            # Backward pass with shape validation
            dvalues = self.loss.backward(output, y_batch)

            # max_grad = max(
            #     np.max(np.abs(layer.dweights)) 
            #     for layer in model.layers 
            #     if isinstance(layer, Layer_Dense)
            # )
            # print(f"Max gradient: {max_grad:.4f}")
            #dvalues = self.loss.dinputs
            
            # Verify gradient shape matches output
            assert dvalues.shape == output.shape, \
                f"Gradient shape mismatch: {dvalues.shape} vs {output.shape}"
            
            # Propagate gradients
            i = 0
            for layer in reversed(self.model.layers):
                dvalues = layer.backward(dvalues)
                
                # Ensure numpy arrays
                dvalues = to_ndarray(dvalues)
                print(f"{self.epoch}:{i}) weights: {layer.weights}")
                i+=1

            # Update parameters
            self.update_params()
            
            batch_losses.append(loss)
            batch_accuracies.append(accuracy)

        return batch_losses, batch_accuracies


    def update_params(self):
        assert self.model is not None
        assert self.optimizer is not None

        self.optimizer.pre_update_params()
        for layer in self.model.layers:
            if isinstance(layer, Layer_Dense):
                self.optimizer.update_params(layer)
        self.optimizer.post_update_params()
