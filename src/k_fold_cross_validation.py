from train_and_evaluate import *
from ensemble.cascade_correlation import CascadeCorrelation

def k_fold_cross_validation_manual(X, y, params, k=5, seed=42):
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

        model = CascadeCorrelation(input_size = 17, output_size= 1, activation=params['hidden_activation'], output_activation = params['output_activation'], max_depth=params['max_depth'])

        _, val_acc = train_and_evaluate(X_train, y_train, X_val, y_val,
                                        learning_rate=params['learning_rate'],
                                        n_epochs=params['n_epochs'],
                                        batch_size=params['batch_size'], weight_decay=params['weight_decay'], patience=params['patience'],
                                        model=model)

        val_accuracies.append(val_acc)
        print(f"✅ Fold {fold+1}/{k} | Validation Accuracy: {val_acc:.4f}")
        current = stop

    mean_accuracy = np.mean(val_accuracies)
    print(f"\n📊 Manual K-Fold | Mean Validation Accuracy over {k} folds: {mean_accuracy:.4f}")
    return val_accuracies, mean_accuracy