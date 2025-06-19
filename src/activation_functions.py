import numpy as np

class Activation:
    def forward(self, inputs):
        raise NotImplementedError

    def backward(self, dvalues):
        raise NotImplementedError

class Activation_Linear(Activation):
    """Linear activation for regression output"""
    def forward(self, inputs):
        self.inputs = inputs
        self.output = inputs.copy()
        
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()

class Activation_ReLU(Activation):
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        # Since we need to modify the orginial variable,
        # let's make a copy of the values first
        self.dinputs = dvalues.copy()
        
        # zero gradient where input values were negative
        self.dinputs[self.inputs <= 0] = 0

class Activation_Leaky_ReLU(Activation):
    def forward(self, inputs, alpha=0.01):
        self.inputs = inputs
        self.output = np.where(inputs > 0, inputs, alpha * inputs)

    def backward(self, dvalues, alpha=0.01):  # f′(x) = {1, α : if  x > 0 ; if x ≤ 0}
        self.dinputs = dvalues.copy()
        self.dinputs = np.where(self.inputs > 0, 1, alpha)
        self.dinputs *= dvalues

class Activation_Sigmoid(Activation):
    def forward(self, inputs):
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))
    
    def backward(self, dvalues):
        # Compute derivative of sigmoid
        sigmoid_derivative = self.output * (1 - self.output)
        # Multiply by incoming gradients (chain rule)
        self.dinputs = dvalues * sigmoid_derivative

class Activation_Tanh(Activation):
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.tanh(inputs)

    def backward(self, dvalues):  # f′(x) = 1−tanh**2(x)
        self.dinputs = dvalues.copy()
        self.dinputs = 1 - (self.output)**2
        self.dinputs *= dvalues

class Activation_ELU(Activation):
    def forward(self, inputs, alpha=1.0):   
        self.inputs = inputs 
        self.output = np.where(inputs > 0, inputs, alpha * np.exp(inputs) - 1)
        
    def backward(self, dvalues, alpha=1.0):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs < 0] = dvalues[self.inputs < 0] * (self.output + alpha)
        self.dinputs 
    
    
class Activation_Softmax(Activation):
    def forward(self, inputs, training=True):
        # Store inputs for backward pass
        self.inputs = inputs
        
        # Get unnormalized probabilities by subtracting max for numerical stability
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        
        # Normalize for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        
        # Add epsilon to avoid zero probabilities
        epsilon = 1e-7
        self.output = np.clip(probabilities, epsilon, 1 - epsilon)
        
        return self.output

    def backward(self, dvalues):
        # Create uninitialized array
        self.dinputs = np.empty_like(dvalues)
        
        # Enumerate outputs and gradients
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            # Reshape output array
            single_output = single_output.reshape(-1, 1)
            
            # Calculate Jacobian matrix of the output
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            
            # Calculate sample-wise gradient
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)
        
        return self.dinputs