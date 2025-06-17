import numpy as np

from src.neural_network import NN
from src.ensemble.cascade_correlation import CascadeCorrelation
from src.activation_functions import *
from src.train_and_evaluate import Train


def k_fold_cross_validation_manual(X, y, hyperparams: dict, k=5, seed=42, regression=False):
    assert 'l1' in hyperparams, "K-Fold Cross valid no l1"
    assert 'l2' in hyperparams, "K-Fold Cross valid no l2"
    assert 'hidden_size' in hyperparams, "K-Fold Cross valid no hidden_size"
    assert 'hidden_activation' in hyperparams, "K-Fold Cross valid no hidden_activation"
    assert 'dropout_rate' in hyperparams, "K-Fold Cross valid no dropout_rate"
    assert 'batch_norm' in hyperparams, "K-Fold Cross valid no batch_norm"
    assert 'CC' in hyperparams, "K-Fold Cross valid no CC"

    if regression:
        input_size = 12
        output_size = 3
    else:
        input_size = 17
        output_size = 1

    np.random.seed(seed)
    n_samples = len(X)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    fold_sizes = np.full(k, n_samples // k)
    fold_sizes[:n_samples % k] += 1  # Distribute the remainder
    current = 0

    val_accuracies = []

    for fold in range(k):
        start, stop = current, current + fold_sizes[fold]
        val_indices = indices[start:stop]
        train_indices = np.concatenate([indices[:start], indices[stop:]])

        X_train, y_train = X[train_indices], y[train_indices]
        X_val, y_val = X[val_indices], y[val_indices]

        # backward compatible fix
        if 'hidden_activation' in hyperparams and isinstance(hyperparams['hidden_activation'],list):
            hyperparams['hidden_activation'] = hyperparams['hidden_activation'][0]

        if hyperparams['CC']:
            model = CascadeCorrelation(input_size = 17, output_size= 1, activation=hyperparams['hidden_activation'], output_activation = Activation_Sigmoid)
        else:
            model = NN(
                l1=hyperparams['l1'],
                l2=hyperparams['l2'],
                input_size=input_size,
                hidden_size=hyperparams['hidden_size'],
                output_size=output_size,
                hidden_activation=hyperparams['hidden_activation'],
                dropout_rate=hyperparams['dropout_rate'],
                use_batch_norm=hyperparams['batch_norm'],
                weights_init=hyperparams['weights_init'],
                n_h_layers=hyperparams['n_h_layers']
            )

        train = Train(hyperparams, model, regression)
        _, val_acc = train.train_and_evaluate(X_train, y_train, X_val, y_val)

        val_accuracies.append(val_acc)
        print(f"✅ Fold {fold+1}/{k} | Validation Accuracy: {val_acc:.4f}")
        current = stop

    mean_accuracy = np.mean(val_accuracies)
    print(f"\n📊 Manual K-Fold | Mean Validation Accuracy over {k} folds: {mean_accuracy:.4f}")
    return val_accuracies, mean_accuracy