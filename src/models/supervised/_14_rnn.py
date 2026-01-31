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
        y_pred = self._forward_propagation(X, n_samples, timesteps)
        
        self._backward_propagation(X, y, y_pred, timesteps)

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

    def _forward_propagation(self, X: np.ndarray, n_samples: int, timesteps: int) -> None:
        """Perform forward propagation through the network."""
        self.cache = {'A_0': X}
        self.cache['H_0'] = np.zeros((n_samples, self.hidden_layers[0]))
        for t in range(timesteps):
            for i in range(len(self.hidden_layers)):
                Z = self.cache[f'A_0'][:, t, :] @ self.W[f'W_{i}'] + self.cache[f'H_{i}'] @ self.Wh[f'Wh_{i}'] + self.b[f'b_{i}']
                self.cache[f'Z_{i+1}'] = Z
                self.cache[f'A_{i+1}'] = self._activation_func(Z, method=self.hidden_activation)
                self.cache[f'H_{i}'] = self.cache[f'A_{i+1}']

            Z_out = self.cache[f'A_{len(self.hidden_layers)}'] @ self.W['W_out'] + self.b['b_out']
            self.cache['Z_out'] = Z_out
            self.cache['A_out'] = self._activation_func(Z_out, method=self.output_activation)

        return self.cache['A_out']

    def _backward_propagation(self, X: np.ndarray, y: np.ndarray, y_pred: np.ndarray, timesteps: int) -> None:
        """Perform backward propagation to update weights and biases."""
        grad_A_out = -2*(y - y_pred) / len(y)
        grad_Z_out = grad_A_out * self._activation_derivative(self.cache['Z_out'], method=self.output_activation)

        grad_W_out = self.cache['A_out'].T @ grad_Z_out / len(y)
        grad_b_out = np.sum(grad_Z_out, axis=0, keepdims=True) / len(y)
        self.W['W_out'] = self.W['W_out'] - self.lr * grad_W_out
        self.b['b_out'] = self.b['b_out'] - self.lr * grad_b_out

        grad_A = grad_Z_out @ self.W['W_out'].T / len(y)
        grad_H = np.zeros_like(grad_A)
        for t in range(timesteps-1, -1, -1):
            for i in range(len(self.hidden_layers)-1, -1, -1):
                grad_Z = (grad_A + grad_H) * self._activation_derivative(self.cache[f'Z_{i+1}'], method=self.hidden_activation)

                grad_W = self.cache[f'A_0'][:, t, :].T @ grad_Z / len(y)
                grad_Wh = self.cache[f'H_{i}'].T @ grad_Z / len(y)
                grad_b = np.sum(grad_Z, axis=0, keepdims=True) / len(y)

                self.W[f'W_{i}'] = self.W[f'W_{i}'] - self.lr * grad_W
                self.Wh[f'Wh_{i}'] = self.Wh[f'Wh_{i}'] - self.lr * grad_Wh
                self.b[f'b_{i}'] = self.b[f'b_{i}'] - self.lr * grad_b

                grad_A = grad_Z @ self.W[f'W_{i}'].T / len(y)
                grad_H = grad_Z @ self.Wh[f'Wh_{i}'].T / len(y)

                
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
