import pandas as pd 
import numpy as np 

class NeuralNetwork:
    """Neural Network classifier from scratch."""
    
    def __init__(self, hidden_layers: list[int]=[10, 10], learning_rate: float=0.01, epochs: int=1000, tol: float=1e-5):
        """Initialize hyperparameters for Neural Network."""
        self.hidden_layers = hidden_layers
        self.lr = learning_rate 
        self.epochs = epochs
        self.tol = tol

        self.cache = {}
        self.W = {}
        self.b = {}
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'NeuralNetwork':
        """Train the Neural Network classifier."""
        X = X.values if isinstance(X, pd.DataFrame) else X
        y = y.values if isinstance(y, pd.Series) else y
        n_features = X.shape[1]

        def loss_function(y: np.ndarray, y_pred: np.ndarray) -> float:
            """Mean Squared Error loss function."""
            return np.mean((y - y_pred) ** 2)

        self._initialize_weights(n_features)
        list_loss = [np.inf]
        for epoch in range(self.epochs):
            self._forward_propagation(X)
            y_pred = self.cache['A_out']

            self._backward_propagation(X, y, y_pred)

            loss = loss_function(y, y_pred)
            list_loss.append(loss)
            if np.abs(list_loss[-1] - list_loss[-2]) < self.tol:
                break

            if epoch % 100 == 0:
                print(f"Epoch {epoch+1}/{self.epochs}, Loss: {loss:.4f}")

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class labels for the input data."""
        X = X.values if isinstance(X, pd.DataFrame) else X
        self._forward_propagation(X)
        y_pred = self.cache['A_out'].flatten()
        return np.where(y_pred >= 0.5, 1, 0)

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
        grad_b_out = -(y.reshape(-1, 1) - y_pred)/len(y)
        grad_W_out = self.cache[f'A_{len(self.hidden_layers)}'].T/len(y) @ grad_b_out
        self.W['W_out'] = self.W['W_out'] - self.lr * grad_W_out
        self.b['b_out'] = self.b['b_out'] - self.lr * np.sum(grad_b_out, axis=0, keepdims=True)
        
        grad_A = grad_b_out @ self.W['W_out'].T/len(y)
        grad_Z = grad_A * self._activation_derivative(self.cache[f'Z_out'], method='relu')/len(y)
        grad_W = self.cache[f'A_{len(self.hidden_layers)-1}'].T/len(y) @ grad_Z
        grad_b = np.sum(grad_Z, axis=0, keepdims=True)/len(y)
        self.W[f'W_{len(self.hidden_layers)-1}'] = self.W[f'W_{len(self.hidden_layers)-1}'] - self.lr * grad_W
        self.b[f'b_{len(self.hidden_layers)-1}'] = self.b[f'b_{len(self.hidden_layers)-1}'] - self.lr * grad_b
        for i in range(len(self.hidden_layers)-2, -1, -1):
            grad_A = grad_Z @ self.W[f'W_{i+1}'].T/len(y)
            grad_Z = grad_A * self._activation_derivative(self.cache[f'Z_{i+1}'], method='relu')/len(y)
            grad_W = self.cache[f'A_{i}'].T/len(y) @ grad_Z
            grad_b = np.sum(grad_Z, axis=0, keepdims=True)/len(y)
            self.W[f'W_{i}'] = self.W[f'W_{i}'] - self.lr * grad_W
            self.b[f'b_{i}'] = self.b[f'b_{i}'] - self.lr * grad_b

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
