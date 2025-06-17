import csv
import random

from k_fold_cross_validation import *
from neural_network import *
from train_and_evaluate import *


def random_search(X_train, y_train, param_distributions, n_iters, csv_path="top_5_results.csv", regression=False):
    results = []
    
    for iter in range(n_iters):
        params = {
            key: random.choice(values)
            for key, values in param_distributions.items()
        }

        # Train and evaluate the model
        print(f"Iteration {iter}")
        _, val_accuracy = k_fold_cross_validation_manual(X=X_train, y=y_train, hyperparams=params, k=5, seed=42, regression=regression)

        # Save the result
        result = params.copy()
        result["val_accuracy"] = val_accuracy
        results.append(result)

    # Sort by validation accuracy (descending)
    top_results = sorted(results, key=lambda x: x["val_accuracy"], reverse=True)[:]

    # Save top 5 to CSV
    fieldnames = list(top_results[0].keys())

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(top_results)

    # Return best of top 5
    best = top_results[0]
    return best, best["val_accuracy"]


