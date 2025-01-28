import numpy as np

class BatchNormalization:
    def __init__(self, epsilon=1e-7, momentum=0.9):
        self.epsilon = epsilon  # Small constant for numerical stability
        self.momentum = momentum  # Momentum for running averages
        
        # Parameters to be learned
        self.gamma = None  # Scale parameter
        self.beta = None   # Shift parameter
        
        # Running estimates for inference
        self.running_mean = None
        self.running_var = None
        
        # Cache for backward pass
        self.cache = {}
        
        # Gradients
        self.dgamma = None
        self.dbeta = None
        
    def forward(self, inputs, training=True):
        if self.gamma is None:
            self.gamma = np.ones(inputs.shape[1])
        if self.beta is None:
            self.beta = np.zeros(inputs.shape[1])
        if self.running_mean is None:
            self.running_mean = np.zeros(inputs.shape[1])
        if self.running_var is None:
            self.running_var = np.ones(inputs.shape[1])
            
        if training:
            # Mini-batch statistics
            batch_mean = np.mean(inputs, axis=0)
            batch_var = np.var(inputs, axis=0)
            
            # Normalize
            x_norm = (inputs - batch_mean) / np.sqrt(batch_var + self.epsilon)
            
            # Update running estimates
            self.running_mean = (self.momentum * self.running_mean + 
                               (1 - self.momentum) * batch_mean)
            self.running_var = (self.momentum * self.running_var + 
                              (1 - self.momentum) * batch_var)
            
            # Cache values for backward pass
            self.cache.update({
                'x_norm': x_norm,
                'batch_mean': batch_mean,
                'batch_var': batch_var,
                'inputs': inputs,
            })
            
        else:
            # Use running estimates for inference
            x_norm = ((inputs - self.running_mean) / 
                     np.sqrt(self.running_var + self.epsilon))
        
        # Scale and shift
        self.output = self.gamma * x_norm + self.beta
        return self.output
    
    def backward(self, dvalues):
        # Get cached values
        x_norm = self.cache['x_norm']
        batch_mean = self.cache['batch_mean']
        batch_var = self.cache['batch_var']
        inputs = self.cache['inputs']
        
        # Get batch size
        N = dvalues.shape[0]
        
        # Gradients for gamma and beta
        self.dgamma = np.sum(dvalues * x_norm, axis=0)
        self.dbeta = np.sum(dvalues, axis=0)
        
        # Gradient with respect to normalized inputs
        dx_norm = dvalues * self.gamma
        
        # Gradient with respect to variance
        dvar = np.sum(dx_norm * (inputs - batch_mean) * 
                     -0.5 * (batch_var + self.epsilon)**(-1.5), axis=0)
        
        # Gradient with respect to mean
        dmean = np.sum(dx_norm * -1/np.sqrt(batch_var + self.epsilon), axis=0) + \
                dvar * np.mean(-2 * (inputs - batch_mean), axis=0)
        
        # Gradient with respect to inputs
        self.dinputs = dx_norm / np.sqrt(batch_var + self.epsilon) + \
                      dvar * 2 * (inputs - batch_mean) / N + \
                      dmean / N
        
        return self.dinputs
    
    def get_parameters(self):
        return {'gamma': self.gamma, 'beta': self.beta}
    
    def set_parameters(self, parameters):
        self.gamma = parameters['gamma'].copy()
        self.beta = parameters['beta'].copy()