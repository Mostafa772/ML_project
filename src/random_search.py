# Random search for finding the best hyperparameters
def random_search(param_distributions, n_iters):
    best_hyperparams = None
    best_performance = -np.inf
    
    for _ in range(n_iters):
        # Let's have sample hyperparameters from distributions
        params = {
            'learning_rate': random.choice(param_distributions['learning_rate']),
            'l1': random.choice(param_distributions['l1']),
            'l2': random.choice(param_distributions['l2']),
            'dropout_rate': random.choice(param_distributions['dropout_rate']),
            'batch_size': random.choice(param_distributions['batch_size']),
            'n_epochs': random.choice(param_distributions['n_epochs']),
            'activation': random.choice(param_distributions['activation']),
        }
        # We train and evaluate the model
        val_accuracy = train_and_evaluate(**params)
        
        # Update the hyperparamters if the current model is doing great
        if val_accuracy > best_performance:
            best_performance = val_accuracy
            best_hyperparams = params
    return best_hyperparams, best_performance