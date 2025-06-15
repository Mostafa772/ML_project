import csv
import random

from k_fold_cross_validation import *
from neural_network import *
from train_and_evaluate import *


# # Random search for finding the best hyperparameters


# param_distributions = {
#     'hidden_size': [[3], [4], [5], [6]],
#     'hidden_activation': [[Activation_Tanh], [Activation_Leaky_ReLU], [Activation_Sigmoid], [Activation_ReLU]],
#     'batch_norm': [[True], [False]],
#     'learning_rate': [1e-4, 1e-3, 1e-2, 1e-1],
#     'l1': [0.0, 1e-5, 1e-4, 1e-3, 1e-2],
#     'l2': [0.0, 1e-5, 1e-4, 1e-3, 1e-2],
#     'dropout_rate': [0.0, 0.1, 0.3],
#     'batch_size': [1000],
#     'n_epochs': [150, 200],
#     'weight_decay': list(np.arange(0.0, 0.05, 0.01)),
#     'patience': [5, 10, 15]
# }



def random_search(X_train, y_train, param_distributions, n_iters, csv_path="top_5_results.csv"):
    results = []

    for _ in range(n_iters):
        # layers = random.choice(param_distributions["hidden_configs"])
        params = {
            'learning_rate': random.choice(param_distributions['learning_rate']),
            'l1': random.choice(param_distributions['l1']),
            'l2': random.choice(param_distributions['l2']),
            'dropout_rate': random.choice(param_distributions['dropout_rate']),
            'batch_size': random.choice(param_distributions['batch_size']),
            'n_epochs': random.choice(param_distributions['n_epochs']),
            'hidden_size': random.choice(param_distributions["hidden_size"]),
            'hidden_activation': random.choice(param_distributions["hidden_activation"]), 
            'batch_norm': random.choice(param_distributions["batch_norm"]),
            'weight_decay': random.choice(param_distributions["weight_decay"]),
            'patience': random.choice(param_distributions['patience'])            
        }
        
        # Train and evaluate the model
        _, val_accuracy = k_fold_cross_validation_manual(X=X_train, y=y_train, l1=params["l1"], l2=params["l2"],
                                    hidden_size=params["hidden_size"],
                                    hidden_activation=params["hidden_activation"],
                                    dropout_rate=params["dropout_rate"],
                                    use_batch_norm=params["batch_norm"], 
                                    learning_rate=params["learning_rate"],
                                    n_epochs=params["n_epochs"],
                                    batch_size=params["batch_size"],
                                    weight_decay=params["weight_decay"],
                                    patience=params["patience"],
                                    k=5, seed=42)

        # Save the result
        result = params.copy()
        result["val_accuracy"] = val_accuracy
        result['hidden_activation'] = [str(cls.__name__) for cls in params["hidden_activation"]]
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


