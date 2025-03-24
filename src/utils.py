import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def create_batches(X, y, batch_size: int):
    """Create mini-batches from the data"""
    # Convert to numpy array if input is pandas DataFrame/Series
    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(y, pd.Series):
        y = y.values
    
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_indices = indices[start_idx:end_idx]
        
        yield X[batch_indices], y[batch_indices]


def plot_accuracies(train_vals: np.ndarray, val_vals: np.ndarray, test_accuracy: float, label1="Training accuracies", label2="Validation accuracies", label3="Test accuracy", title="Accuracy Over Epochs"):
    """
    Plot training and validation accuracies over epochs.

    Parameters:
    - train_accuracies (list or array): Accuracy values for training data over epochs.
    - val_accuracies (list or array): Accuracy values for validation data over epochs.
    - title (str): Title of the plot. Default is "Accuracy Over Epochs".
    """
    plt.figure(figsize=(8, 6))
    plt.plot(train_vals, label=label1, color="blue", linewidth=2)
    plt.plot(val_vals, label=label2, linestyle="--", color="orange", linewidth=2)
    plt.axhline(y=test_accuracy, color="red", label=label3, linewidth=0.5)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_losses(train_vals: np.ndarray, val_vals: np.ndarray, test_loss: float, label1="Training loss", label2="Validation loss", label3 = "Test loss", title="Loss Over Epochs"):
    """
    Plot training and validation losses over epochs.

    Parameters:
    - train_losses (list or array): Loss values for training data over epochs.
    - val_losses (list or array): Loss values for validation data over epochs.
    - title (str): Title of the plot. Default is "Loss Over Epochs".
    """
    plt.figure(figsize=(8, 6))
    plt.plot(train_vals, label=label1, color="blue", linewidth=2)
    plt.plot(val_vals, label=label2, linestyle="--", color="orange", linewidth=2)
    plt.axhline(y=test_loss, color="red", label=label3, linewidth=0.5)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

def to_ndarray(array: pd.DataFrame | pd.Series | np.ndarray) -> np.ndarray:
    if isinstance(array, np.ndarray):
        return array

    if isinstance(array, pd.DataFrame):
        return array.values
    
    if isinstance(array, pd.Series):
        return array.array.to_numpy()