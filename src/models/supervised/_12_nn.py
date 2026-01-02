import pandas as pd 
import numpy as np 

class NeuralNetwork:
    """Neural Network classifier from scratch."""
    
    def __init__(self, hidden_layers: list[int]=[10, 10], learning_rate: float=0.01, epochs: int=1000):
        """Initialize hyperparameters for Neural Network."""
        self.hidden_layers = hidden_layers
        self.lr = learning_rate 
        self.epochs = epochs

        self.cache = {}
        self.W = {}
        self.b = {}
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'NeuralNetwork':
        """Train the Neural Network classifier."""
        X.values if isinstance(X, pd.DataFrame) else X
        y.values if isinstance(y, pd.Series) else y
        n_features = X.shape[1]

        self._initialize_weights(n_features)
        for epoch in range(self.epochs):
            self._forward_propagation(X)
            y_pred = self.cache['A_out']

            self._backward_propagation(X, y, y_pred)

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class labels for the input data."""
        pass

    def _initialize_weights(self, n_features: int) -> None:
        """Initialize weights and biases for the network."""
        self.W, self.b = {}, {}
        prev_layer_size = n_features
        for i in range(len(self.hidden_layers)):
            self.W[f'W_{i}'] = np.random.randn(prev_layer_size, self.hidden_layers[i]) * np.sqrt(2 / prev_layer_size)
            self.b[f'b_{i}'] = np.zeros((1, self.hidden_layers[i]))
            prev_layer_size = self.hidden_layers[i]

        self.W['W_out'] = np.random.randn(self.hidden_layers[-1], 1) * np.sqrt(2 / prev_layer_size)
        self.b['b_out'] = np.zeros((1, 1))

    def _forward_propagation(self, X: np.ndarray) -> None:
        """Perform forward propagation."""
        self.cache = {'A_0': X}
        for i in range(len(self.hidden_layers)):
            Z = self.cache[f'A_{i}'] @ self.W[f'W_{i}'] + self.b[f'b_{i}']
            self.cache[f'Z_{i+1}'] = Z
            self.cache[f'A_{i+1}'] = self._activation_func(Z, method='relu')

        Z_out = self.cache[f'A_{len(self.hidden_layers)}'] @ self.W['W_out'] + self.b['b_out']
        self.cache['Z_out'] = Z_out
        self.cache['A_out'] = self._activation_func(Z_out, method='relu')

    def _backward_propagation(self, X: np.ndarray, y: np.ndarray, y_pred: np.ndarray) -> None:
        """Perform backward propagation and update weights."""

        pass

    def _activation_func(self, y: np.ndarray, method='relu') -> np.ndarray:
        """ReLU activation function."""
        if method == 'relu':
            return np.maximum(0, y) 
        return

    def _activation_derivative(self, y: np.ndarray, method='relu') -> np.ndarray:
        """Derivative of ReLU activation function."""
        if method == 'relu':
            return np.where(y > 0, 1, 0)
        return
