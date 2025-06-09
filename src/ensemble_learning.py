from loss_functions import *
from train_and_evaluate import *
from data_preprocessing import *
from data_split import *
from model import *


class EnsembleNN:
    def __init__(self, n_models=5):
        self.models = []
        self.n_models = n_models
        self.loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

    def create_and_train_models(self, hyperparams):
        # Create and train multiple models with the same hyperparameters
        for i in range(self.n_models):
            print(f"Training model {i+1}/{self.n_models}")
            # Train model using existing train_and_evaluate function
            model, val_accuracy = train_and_evaluate(
                learning_rate=hyperparams['learning_rate'],
                # l1=hyperparams['l1'],
                # l2=hyperparams['l2'],
                # dropout_rate=hyperparams['dropout_rate'],
                batch_size=hyperparams['batch_size'],
                n_epochs=hyperparams['n_epochs'],
                weight_decay=hyperparams['weight_decay'],
                # model=hyperparams['model']
                # activation=hyperparams['activation']
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                model=model,
            )
            self.models.append(model)
            print(f"Model {i+1} validation accuracy: {val_accuracy:.4f}")

    def predict(self, X):
        """Make predictions using majority voting"""
        predictions = []
        for model in self.models:
            model.forward(X, training=False)
            self.loss_activation.forward(
                model.output, np.zeros((X.shape[0], 2)))  # Dummy y values
            pred = np.argmax(self.loss_activation.output, axis=1)
            predictions.append(pred)

        # Majority voting
        predictions = np.array(predictions)
        final_predictions = np.apply_along_axis(
            lambda x: np.bincount(x).argmax(),
            axis=0,
            arr=predictions
        )
        return final_predictions
    
