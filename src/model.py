from neural_network import *
# from ensemble_learning import *
from random_search import *

best_params, best_acc = random_search(
    X_train, y_train, param_distributions, n_iters=10)

model = NN(
    l1=best_params['l1'],
    l2=best_params['l2'],
    input_size=17,
    hidden_sizes=best_params['hidden_size'],
    output_size=1,
    hidden_activations=best_params['hidden_activation'],
    dropout_rates=best_params['dropout_rate'],
    use_batch_norm=best_params['batch_norm'],
)
batch_size = batch_size
learning_rate = learning_rate
n_epochs = n_epochs
