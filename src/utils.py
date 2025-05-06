import operator
import os
import pickle
from functools import reduce

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def normalize(data, type="minmax"):
    data_normalized = data.copy()
    # Select numeric columns only for normalization
    numeric_data = data_normalized.select_dtypes(include=["float64", "int64"])
    if type=="minmax":
        # Calculate min and max from the training data
        min_vals = numeric_data.min()
        max_vals = numeric_data.max()

        # Normalize each numeric column separately using the min-max formula
        normalized_data = (numeric_data - min_vals) / (max_vals - min_vals)

    if type=="zscore" or type=="z-score":
        means = numeric_data.mean(axis=0)
        stds = numeric_data.std(axis=0)
        # Avoid division by zero for constant columns
        # stds_replaced = stds.replace(0, 1)
        # Apply Z-score normalization
        normalized_data = (numeric_data - means) / stds
    
    
    # Rejoin with non-numeric columns if needed
    non_numeric_data = data.select_dtypes(exclude=["float64", "int64"])
    final_data = pd.concat([non_numeric_data, normalized_data], axis=1)

    return final_data
    
    
def create_batches(X, y, batch_size):
    """Create mini-batches from the data"""
    # Convert to numpy array if input is pandas DataFrame/Series
    # if isinstance(X, pd.DataFrame):
    #     X = X.values
    # if isinstance(y, pd.Series):
    #     y = y.values
    X = np.array(X)
    y = np.array(y).reshape(-1, 1)
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
 


def plot_accuracies(train_vals: np.ndarray, val_vals: np.ndarray, test_accuracy: float, label1="Training accuracies", label2="Validation accuracies", label3="Test accuracy", title="Accuracy Over Epochs", filename: str| None = None):
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
    if filename:
        with open(filename, "wb") as file:
            plt.savefig(file)
    else:
        plt.show()


def plot_losses(train_vals: np.ndarray, val_vals: np.ndarray, test_loss: float, label1="Training loss", label2="Validation loss", label3 = "Test loss", title="Loss Over Epochs", filename: str| None = None):
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
    if filename:
        with open(filename, "wb") as file:
            plt.savefig(file)
    else:
        plt.show()

def save_results(dataset: str, model, best_hyperparams, metrics, save_accuracies=False):
    "Saves model weights, hyperparams and results in the /data/{dataset} directory"
    pwd = os.curdir
    directory = f"{pwd}/data/{dataset}/"
    try:
        os.mkdir(directory)
    except:
        pass

    print(directory)
    # Save model weights
    with open(f"{directory}weights.bin", "wb") as file:
        np.save(file, model.layers)
    
    # Save best found hyperparameters
    with open(f"{directory}hyperparams.bin", "wb") as file:
        pickle.dump(best_hyperparams, file)

    # Save metrics
    with open(f"{directory}metrics.bin", "wb") as file:
        pickle.dump(metrics, file)

    # Save Graphs
    plot_losses(metrics["train_loss"], metrics["validation_loss"], metrics["test_loss"], filename = f"{directory}losses.png")
    if save_accuracies:
        plot_accuracies(metrics["train_accuracy"], metrics["validation_accuracy"], metrics["test_accuracy"], filename = f"{directory}accuracy.png")



def count_permutations(param_grid):
    """
    Counts the number of all possible permutations from a hyperparameter grid.
    
    Parameters:
    - param_grid (dict): Dictionary where keys are hyperparameter names and values are numpy arrays of options.
    
    Returns:
    - int: Total number of possible permutations.
    """
    sizes = [len(values) for values in param_grid.values()]
    return reduce(operator.mul, sizes, 1)

def train_val_split(X, y, val_ratio=0.2, seed=42, shuffle=True):
    """
    Manually split X and y into training and validation sets.

    Parameters:
    - X: numpy array of features
    - y: numpy array of labels
    - val_ratio: fraction of data to use for validation
    - seed: random seed for reproducibility
    - shuffle: whether to shuffle before splitting

    Returns:
    - X_train, X_val, y_train, y_val
    """
    
    
    assert len(X) == len(y), "X and y must have the same number of samples"
    if not isinstance(X, np.ndarray):
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)
    n_samples = len(X)
    n_val = int(n_samples * val_ratio)

    indices = np.arange(n_samples)
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(indices)

    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    X_train = X[train_indices]
    y_train = y[train_indices]
    X_val = X[val_indices]
    y_val = y[val_indices]

    return X_train, X_val, y_train, y_val





# def train_test_split(data, train_percent=80, target=None):
    
#     """
#     Splits the data into training and testing sets.

#     Parameters:
#     - data (DataFrame): The input data to be split.
#     - train_percent (int): The percentage of data to be used for training. Default is 80.

#     Returns:
#     - X_train (DataFrame): Training features.
#     - X_test (DataFrame): Testing features.
#     - y_train (Series): Training labels.
#     - y_test (Series): Testing labels.
#     """
#     # Shuffle the data
#     shuffled_data = data.sample(frac=1, random_state=42).reset_index(drop=True)
    
#     # Calculate the split index
#     split_index = int(len(shuffled_data) * train_percent / 100)
    
#     # Split the data into training and testing sets
#     train_data = shuffled_data[:split_index]
#     test_data = shuffled_data[split_index:]
    
#     # Separate features and labels
#     X_train = train_data.drop(columns=target)
#     y_train = train_data[target]

#     X_test = test_data.drop(columns=target)
#     y_test = test_data[target]
    
#     return X_train, X_test, y_train, y_test
    