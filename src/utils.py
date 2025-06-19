import operator
import os
import pickle
from functools import reduce

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def normalize(data, type="minmax", params=None):
    # Ensure input is a DataFrame
    data = pd.DataFrame(data)
    
    # Separate numeric and non-numeric columns
    numeric_data = data.select_dtypes(include=["float64", "int64", "float32", "int32"])
    non_numeric_data = data.select_dtypes(exclude=["float64", "int64", "float32", "int32"])

    if numeric_data.empty:
        raise ValueError("No numeric columns found to normalize.")

    if type == "minmax":
        if params is None:
            min_vals = numeric_data.min()
            max_vals = numeric_data.max()
        else:
            min_vals, max_vals = params
        normalized_data = (numeric_data - min_vals) / (max_vals - min_vals)
        scaler_params = (min_vals, max_vals)

    elif type in ("zscore", "z-score"):
        if params is None:
            means = numeric_data.mean()
            stds = numeric_data.std()
        else:
            means, stds = params
        normalized_data = (numeric_data - means) / stds
        scaler_params = (means, stds)

    else:
        raise ValueError(f"Unknown normalization type: {type}")

    if not non_numeric_data.empty:
        final_data = pd.concat([non_numeric_data.reset_index(drop=True), normalized_data.reset_index(drop=True)], axis=1)
    else:
        final_data = normalized_data

    return final_data, scaler_params
    
    
def create_batches(X, y, batch_size):
    """Create mini-batches from the data (supports multi-output y)"""
    X = np.array(X)
    y = np.array(y)  # Don't reshape — keep the original shape

    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_indices = indices[start_idx:end_idx]
        yield X[batch_indices], y[batch_indices]
 


def plot_scores(train_vals: np.ndarray, val_vals: np.ndarray, test_accuracy: float | None, label1="Training accuracies", label2="Validation accuracies", label3="Test accuracy", title="Accuracy Over Epochs", filename: str| None = None):
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
    if test_accuracy:
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


def plot_losses(train_vals: np.ndarray, val_vals: np.ndarray, test_loss: float | None, label1="Training loss", label2="Validation loss", label3 = "Test loss", title="Loss Over Epochs", filename: str| None = None):
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
    if test_loss:
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
        plot_scores(metrics["train_accuracy"], metrics["validation_accuracy"], metrics["test_accuracy"], filename = f"{directory}accuracy.png")



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
    
    if not isinstance(y, np.ndarray):
        y = np.array(y)

    # Only reshape if y is 1D (single target)
    if y.ndim == 1:
        y = y.reshape(-1, 1)

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





def train_test_split(data, target=None, train_percent=80):
    
    """
    Splits the data into training and testing sets.

    Parameters:
    - data (DataFrame): The input data to be split.
    - train_percent (int): The percentage of data to be used for training. Default is 80.

    Returns:
    - X_train (DataFrame): Training features.
    - X_test (DataFrame): Testing features.
    - y_train (Series): Training labels.
    - y_test (Series): Testing labels.
    """
    # Shuffle the data
    shuffled_data = data.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Calculate the split index
    split_index = int(len(shuffled_data) * train_percent / 100)
    
    # Split the data into training and testing sets
    train_data = shuffled_data[:split_index]
    test_data = shuffled_data[split_index:]
    
    # Separate features and labels
    if target is None:
        return train_data, test_data 
    
    X_train = train_data.drop(columns=target)
    y_train = train_data[target]

    X_test = test_data.drop(columns=target)
    y_test = test_data[target]
    
    return X_train, X_test, y_train, y_test

def preprocess_data(data, target=None, normalize_type="z-score", val_ratio=0.2, regression=False):
    """
    If regression=True:
        - Assumes `data` is a DataFrame with input features + target columns.
        - Automatically extracts and normalizes features and targets.
        - Returns: X_train, X_val, y_train, y_val, X_scaler_params, y_scaler_params

    If regression=False:
        - Returns the shuffled/split version of data.
        - If target is provided, also splits target columns.
    """
    # Shuffle the data
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)

    if regression:
        assert target is not None, "`target` must be specified when regression=True"

        # Ensure target is always a list of columns
        target = [target] if isinstance(target, str) else target
        
        # Split features and targets
        y = data[target]
        X = data.drop(columns=target)
        # print(y)
        # Normalize X
        X_normalized, X_scaler_params = normalize(X, type=normalize_type)

        # Split into train/val
        X_train, X_val, y_train, y_val = train_val_split(X_normalized, y, val_ratio=val_ratio)
        X_train, X_val, y_train, y_val = map(pd.DataFrame, (X_train, X_val, y_train, y_val))
        print(y_val, y_train)
        # Normalize y (with shared parameters)
        y_train_normalized, y_scaler_params = normalize(y_train, type=normalize_type)
        y_val_normalized, _ = normalize(y_val, type=normalize_type, params=y_scaler_params)

        return (
            np.array(X_train),
            np.array(X_val),
            np.array(y_train_normalized),
            np.array(y_val_normalized),
            X_scaler_params,
            y_scaler_params
        )

    else:
        if target is None:
            # Just split the DataFrame
            split_index = int(len(data) * (1 - val_ratio))
            return data[:split_index], data[split_index:]
        else:
            target = [target] if isinstance(target, str) else target
            X = data.drop(columns=target)
            y = data[target]
            split_index = int(len(data) * (1 - val_ratio))
            return (
                X[:split_index], X[split_index:],
                y[:split_index], y[split_index:]
            )
