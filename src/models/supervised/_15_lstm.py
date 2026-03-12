import pandas as pd
import numpy as np

class LSTM:
    """Long Short-Term Memory (LSTM) classifier from scratch."""

    def __init__(self, hidden_layers=[10], learning_rate=0.01, epochs=1000, batch_size=32, tol=1e-5):
        """Initialize hyperparameters for LSTM."""
        self.hidden_layers = hidden_layers
        self.lr = learning_rate 
        
        self.epochs = epochs
        self.batch_size = batch_size
        self.tol = tol

        self.cache = {}
        self.W = {}
        self.H = {}
        self.b = {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'LSTM':
        """Train the LSTM classifier."""
        X = X.values if isinstance(X, pd.DataFrame) else X
        y = y.values if isinstance(y, (pd.Series, pd.DataFrame)) else y
        n_samples, timesteps, n_features = X.shape

        self._initialize_weights(n_features)
        list_loss = [np.inf]

        y_pred = self._forward_propagation(X)
        self._backward_propagation(X, y, y_pred)

        # y_pred = self._forward_propagation(X, n_samples, timesteps)
        
        # self._backward_propagation(X, y, y_pred, timesteps)

        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class labels for the input data."""
        return np.zeros(X.shape[0])
    
    def _initialize_weights(self, n_features: int) -> None:
        """Initialize weights and biases for the network."""
        self.W, self.H, self.b = {}, {}, {}
        prev_layer_size = n_features
        for i in range(len(self.hidden_layers)):
            h_size = self.hidden_layers[i]
            scale_x = np.sqrt(2 / prev_layer_size)
            for gate in ['f', 'i', 'o', 'g']:
                self.W[f'W{gate}_{i}'] = np.random.randn(prev_layer_size, h_size) * scale_x
                self.H[f'H{gate}_{i}'] = np.random.randn(h_size, h_size) * scale_x
                self.b[f'b{gate}_{i}'] = np.zeros((1, h_size))
            
            prev_layer_size = h_size
        
        self.W['W_out'] = np.random.randn(prev_layer_size, 1) * np.sqrt(2 / prev_layer_size)
        self.b['b_out'] = np.zeros((1, 1))

    def _forward_propagation(self, X: np.ndarray) -> np.ndarray:
        """Perform forward propagation through the network."""
        n_samples, timesteps, n_features = X.shape
        
        H = [np.zeros((n_samples, timesteps, h_size)) for h_size in self.hidden_layers]
        C = [np.zeros((n_samples, timesteps, h_size)) for h_size in self.hidden_layers]

        for i in range(len(self.hidden_layers)):
            for t in range(timesteps):
                # GET FIRST STEP OF EACH SAMPLE
                X_t = X[:, t, :] if i==0 else H[i-1][:, t, :]

                # GET PREVIOUS CELL STATE AND HIDDEN STATE
                C_prev = C[i][:, t-1, :] if t>0 else np.zeros((n_samples, self.hidden_layers[i]))
                H_prev = H[i][:, t-1, :] if t>0 else np.zeros((n_samples, self.hidden_layers[i]))
                
                # PREACTIVATION GATES
                Zf = X_t @ self.W[f'Wf_{i}'] + H_prev @ self.H[f'Hf_{i}'] + self.b[f'bf_{i}']
                Zi = X_t @ self.W[f'Wi_{i}'] + H_prev @ self.H[f'Hi_{i}'] + self.b[f'bi_{i}']
                Zg = X_t @ self.W[f'Wg_{i}'] + H_prev @ self.H[f'Hg_{i}'] + self.b[f'bg_{i}']
                Zo = X_t @ self.W[f'Wo_{i}'] + H_prev @ self.H[f'Ho_{i}'] + self.b[f'bo_{i}']

                # ACTIVATION GATES
                Af = self._sigmoid(Zf)      # FORGET GATE 
                Ai = self._sigmoid(Zi)      # INPUT GATE 
                Ag = np.tanh(Zg)            # INPUT GATE
                Ao = self._sigmoid(Zo)      # OUTPUT GATE

                # CELL STATE AND HIDDEN STATE
                C[i][:, t, :] = C_prev * Af + Ai * Ag
                H[i][:, t, :] = Ao * np.tanh(C[i][:, t, :])

        # OUTPUT LAYER
        Z_out = H[-1][:, -1, :] @ self.W['W_out'] + self.b['b_out']
        A_out = self._sigmoid(Z_out).ravel()

        # CACHE
        for keys in ['Zf', 'Zi', 'Zg', 'Zo', 'Af', 'Ai', 'Ag', 'Ao', 'C', 'H', 'Z_out', 'A_out']:
            self.cache[f'{keys}'] = eval(keys)

        return A_out
    
    def _backward_propagation(self, X: np.ndarray, y: np.ndarray, y_pred: np.ndarray) -> None:
        """Perform backward propagation and update weights."""
        n_samples, timesteps, n_features = X.shape

        y_pred = np.clip(y_pred, 1e-9, 1 - 1e-9) 
        dZ_out = (y_pred - y).reshape(-1, 1) / n_samples
        dW_out = self.cache['H'][-1][:, -1, :].T @ dZ_out
        db_out = np.sum(dZ_out, axis=0, keepdims=True)

        dH = [np.zeros_like((n_samples, h_size)) for h_size in self.hidden_layers]

        print(dH)


        
        print(dW_out)

        return

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-x))

    def _sigmoid_derivative(self, x: np.ndarray) -> np.ndarray:
        """Derivative of sigmoid function."""
        return self._sigmoid(x) * (1 - self._sigmoid(x))
    