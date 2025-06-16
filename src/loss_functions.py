import numpy as np

from activation_functions import Activation_Softmax


class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss


class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        if len(y_true.shape) == 1:
            correct_confidence = y_pred_clipped[range(samples), y_true]

        elif len(y_true.shape) == 2:
            correct_confidence = np.sum(y_pred_clipped * y_true, axis=1) 

        negative_log_likelihoods = np.log(correct_confidence)
        return negative_log_likelihoods

    def backward(self, dvalues, y_true):

        # Number of samples
        samples = len(dvalues)
        # Number of labels in every sample
        # We'll use the first sample to count them
        labels = len(dvalues[0])

        # If labels are sparse, turn them into one-hot vector
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        # Calculate gradient
        self.dinputs = -y_true / dvalues
        # Normalize gradient
        self.dinputs = self.dinputs / samples


class Activation_Softmax_Loss_CategoricalCrossentropy(Loss):

    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()


    def forward(self, inputs, y_true):
        # Output layer's activation function
        self.activation.forward(inputs)

        # Set the output
        self.output = self.activation.output

        return self.loss.calculate(self.output, y_true)

    def backward(self, dvalues, y_true):
        samples = len(dvalues)

        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        self.dinputs = dvalues.copy()

        self.dinputs[range(samples), y_true] -= 1

        self.dinputs = self.dinputs / samples


class MSE(Loss):
    def __init__(self):
        self.dinputs = None
        self.output = None

    def forward(self, y_pred, y_true):
        # Remove the shape condition - always calculate loss
        self.output = np.mean((y_pred - y_true)**2)
        return self.output

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        self.dinputs = 2 * (dvalues - y_true) / samples

class MEE(Loss):
    def __init__(self):
        self.dinputs = 0
        self.loss = 0
        self.output = 0
    
    def forward(self, y_pred, y_true):
        self.output = np.mean(np.sqrt(np.sum((y_pred - y_true)**2, axis=1)))
        return self.output
    
    def backward(self, dvalues, y_true):
        # Number of samples and outputs
        samples = len(dvalues)
        outputs = len(dvalues[0])

        differences = dvalues - y_true
        euclidean_distances = np.sqrt(np.sum(differences**2, axis=1, keepdims=True))
        
        # Avoid division by zero
        euclidean_distances = np.maximum(euclidean_distances, 1e-7)

        self.dinputs = differences / euclidean_distances
        
        # Normalize by number of samples and outputs
        self.dinputs = self.dinputs / (samples * outputs)
        
        return self.dinputs
    
    
def r2_score_per_output(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n_outputs = y_true.shape[1]
    r2s = []

    for i in range(n_outputs):
        ss_res = np.sum((y_true[:, i] - y_pred[:, i]) ** 2)
        ss_tot = np.sum((y_true[:, i] - np.mean(y_true[:, i])) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0.0
        r2s.append(r2)
    return r2s  # List of R² values per output

def r2_score_global(y_true, y_pred):
    # print("y_true shape: ", y_true.shape)
    # print("y_pred shape: ", y_pred.shape)
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot != 0 else 0.0