import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def create_batches(X, y, batch_size):
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
        # if y.shape[1] == 1: # ****changed something****
        #     yield X[batch_indices], y[batch_indices]
        # else:
        #     # yield X[batch_indices], y.iloc[batch_indices] # ****changed something****
        #     yield X[batch_indices], y[batch_indices]
        yield X[batch_indices], y[batch_indices]
 


def plot_accuracies(train_vals, val_vals, label1="train_accuracies", label2="val_accuracies", title="Accuracy Over Epochs"):
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
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_losses(train_vals, val_vals, label1="train_losses", label2="val_losses", title="Loss Over Epochs"):
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
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()
