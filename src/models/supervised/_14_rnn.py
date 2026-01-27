import pandas as pd
import numpy as np
from typing import List

class RNN:
    """Recurrent Neural Network (RNN) classifier from scratch."""
    
    def __init__(self, hidden_layers: List[int]=[10], learning_rate: float=0.01, epochs: int=1000, batch_size: int=32, tol: float=1e-5):
        """Initialize hyperparameters for RNN."""
        self.hidden_layers = hidden_layers
        self.lr = learning_rate 
        self.epochs = epochs
        self.batch_size = batch_size
        self.tol = tol

        self.hidden_activation = 'relu'
        self.output_activation = 'relu'

        self.cache = {}
        self.W = {}
        self.Wh = {}
        self.b = {}

    def fit(self, X: pd.DataFrame, y: pd.Series, ) -> 'RNN':
        """Train the RNN classifier."""
        X = X.values if isinstance(X, pd.DataFrame) else X
        y = y.values if isinstance(y, (pd.Series, pd.DataFrame)) else y
        n_samples, timesteps, n_features = X.shape

        self._initialize_weights(n_features)
        list_loss = [np.inf]
        self._forward_propagation(X, n_samples)

        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class labels for the input data."""
        return np.zeros(X.shape[0])
    
    def _initialize_weights(self, n_features: int) -> None:
        """Initialize weights and biases for the network."""
        self.W, self.b, self.H = {}, {}, {}
        prev_layer_size = n_features
        for i in range(len(self.hidden_layers)):
            self.W[f'W_{i}'] = np.random.randn(prev_layer_size, self.hidden_layers[i]) * np.sqrt(2 / prev_layer_size)
            self.b[f'b_{i}'] = np.zeros((1, self.hidden_layers[i]))
            self.Wh[f'Wh_{i}'] = np.random.randn(self.hidden_layers[i], self.hidden_layers[i]) * np.sqrt(2 / prev_layer_size)
            prev_layer_size = self.hidden_layers[i]

        self.W['W_out'] = np.random.randn(self.hidden_layers[-1], 1) * np.sqrt(2 / prev_layer_size)
        self.b['b_out'] = np.zeros((1, 1))

    def _forward_propagation(self, X: np.ndarray, n_samples: int) -> None:
        """Perform forward propagation through the network."""
        self.cache = {'A_0': X}
        for i in range(len(self.hidden_layers)):
            self.cache[f'A_{i+1}'] = np.zeros((n_samples, self.hidden_layers[i]))
            self.cache[f'Z_{i+1}'] = np.zeros((n_samples, self.hidden_layers[i]))
            self.cache[f'H_{i+1}'] = np.zeros((n_samples, self.hidden_layers[i]))

        for t in range(n_samples): 
            for i in range(len(self.hidden_layers)):
                Z = self.cache[f'A_{i}'][t] @ self.W[f'W_{i}'] + self.cache[f'H_{i+1}'][t] @ self.Wh[f'Wh_{i}'] + self.b[f'b_{i}']
                self.cache[f'Z_{i+1}'][t] = Z
                self.cache[f'A_{i+1}'][t] = self._activation_func(Z, method=self.hidden_activation)
                self.cache[f'H_{i+1}'][t+1] = self.cache[f'A_{i+1}'][t]

        print(self.cache)

        return

    def _backward_propagation(self, X: np.ndarray, y: np.ndarray, y_pred: np.ndarray) -> None:
        """Perform backward propagation to update weights and biases."""
        pass

    def _activation_func(self, y: np.ndarray, method='relu') -> np.ndarray:
        """ReLU activation function."""
        if method == 'relu':
            return np.maximum(0, y)
        if method == 'sigmoid':
            return 1 / (1 + np.exp(-y))

    def _activation_derivative(self, y: np.ndarray, method='relu') -> np.ndarray:
        """Derivative of ReLU activation function."""
        if method == 'relu':
            return np.where(y > 0, 1, 0)
        if method == 'sigmoid':
            sigmoid = 1 / (1 + np.exp(-y))
            return sigmoid * (1 - sigmoid)
