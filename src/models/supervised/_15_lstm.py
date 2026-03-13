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

                # CACHE
                for keys in ['Zf', 'Zi', 'Zg', 'Zo', 'Af', 'Ai', 'Ag', 'Ao', 'C', 'H']:
                    self.cache[f'{keys}_{i}'] = eval(keys)

        # OUTPUT LAYER
        H_out = H[-1][:, -1, :]
        Z_out = H_out @ self.W['W_out'] + self.b['b_out']
        A_out = self._sigmoid(Z_out).ravel()

        # CACHE
        for keys in ['Zf', 'Zi', 'Zg', 'Zo', 'Af', 'Ai', 'Ag', 'Ao', 'C', 'H', 'Z_out', 'A_out']:
            self.cache[f'{keys}'] = eval(keys)

        return A_out
    
    def _backward_propagation(self, X: np.ndarray, y: np.ndarray, y_pred: np.ndarray) -> None:
        """Perform backward propagation and update weights."""
        n_samples, timesteps, n_features = X.shape

        y_pred = np.clip(y_pred, 1e-9, 1 - 1e-9)
        dZ_out = y_pred.reshape(-1, 1) - y                                          # BINARY CROSS-ENTROPY LOSS DERIVATIVE
        dW_out = self.cache['H'][-1][:, -1, :].T @ dZ_out / n_samples               # dL/dW = dL/dZ * dZ/dW_ = H.T @ dZ
        db_out = np.sum(dZ_out, axis=0, keepdims=True) / n_samples                  # dL/db = dL/dZ * dZ/db = sum(dZ) * 1
        dH_out = dZ_out @ self.W['W_out'].T / n_samples                             # dL/dH = dL/dZ * dZ/dH = dZ @ W.T

        dH_next = dH_out
        for i in range(len(self.hidden_layers)-1, -1, -1):
           
            for t in range(timesteps-1, -1, -1):
                # GET FIRST STEP OF EACH SAMPLE
                X_t = X[:, t, :] if i==0 else self.cache['H'][i-1][:, t, :]

                # GET PREVIOUS CELL STATE AND HIDDEN STATE
                C_prev = self.cache['C'][i][:, t-1, :] if t>0 else np.zeros((n_samples, self.hidden_layers[i]))
                H_prev = self.cache['H'][i][:, t-1, :] if t>0 else np.zeros((n_samples, self.hidden_layers[i]))

                # GATE VALUES
                Af = self.cache['Af'][i][:, t, :]
                Ai = self.cache['Ai'][:, t, :]  
                Ag = self.cache['Ag'][:, t, :]
                Ao = self.cache['Ao'][:, t, :]
                C = self.cache['C'][i][:, t, :]

                dH_total = (dH_next if t == timesteps - 1 else np.zeros_like(dH_t)) + dH_t

                # ── CELL STATE GRADIENT (two paths) ───────────────────────────────
                # Path 1: H_t = Ao * tanh(C_t)
                # Path 2: C_{t+1} = Af_{t+1} * C_t  (carried via dC_t)
                dC = dH_total * Ao * (1 - np.tanh(C)**2) + dC_t * Af

                # DERIVATIVE OF ACTIVATION GATES
                dAf = dC * C_prev
                dAi = dC * Ag
                dAg = dC * Ai
                dAo = dH_total * np.tanh(C)

                # DERIVATIVE OF PREACTIVATION GATES
                dZf = dAf * Af * (1 - Af)       # sigmoid'(Zf) = Af * (1 - Af)
                dZi = dAi * Ai * (1 - Ai)       # sigmoid'(Zi) = Ai * (1 - Ai)
                dZg = dAg * (1 - Ag**2)         # tanh'(Zg) = 1 - tanh(Zg)^2 = 1 - Ag^2
                dZo = dAo * Ao * (1 - Ao)       # sigmoid'(Zo) = Ao * (1 - Ao)

                # GRADIENTS FOR WEIGHTS AND BIASES
                dWf = X_t.T @ dZf / n_samples
                dWi = X_t.T @ dZi / n_samples
                dWg = X_t.T @ dZg / n_samples
                dWo = X_t.T @ dZo / n_samples
                dbf = np.sum(dZf, axis=0, keepdims=True) / n_samples
                dbi = np.sum(dZi, axis=0, keepdims=True) / n_samples
                dbg = np.sum(dZg, axis=0, keepdims=True) / n_samples
                dbo = np.sum(dZo, axis=0, keepdims=True) / n_samples
                dHf = H_prev.T @ dZf / n_samples
                dHi = H_prev.T @ dZi / n_samples
                dHg = H_prev.T @ dZg / n_samples
                dHo = H_prev.T @ dZo / n_samples

                # UPDATE WEIGHTS AND BIASES
                self.W[f'Wf_{i}'] -= self.lr * dWf
                self.W[f'Wi_{i}'] -= self.lr * dWi  
                self.W[f'Wg_{i}'] -= self.lr * dWg
                self.W[f'Wo_{i}'] -= self.lr * dWo
                self.b[f'bf_{i}'] -= self.lr * dbf
                self.b[f'bi_{i}'] -= self.lr * dbi
                self.b[f'bg_{i}'] -= self.lr * dbg
                self.b[f'bo_{i}'] -= self.lr * dbo
                self.H[f'Hf_{i}'] -= self.lr * dHf
                self.H[f'Hi_{i}'] -= self.lr * dHi
                self.H[f'Hg_{i}'] -= self.lr * dHg
                self.H[f'Ho_{i}'] -= self.lr * dHo

                # GRADIENT FOR PREVIOUS HIDDEN STATE (carried to next time step)
                dH_t = dZf @ self.H[f'Hf_{i}'].T + dZi @ self.H[f'Hi_{i}'].T + dZg @ self.H[f'Hg_{i}'].T + dZo @ self.H[f'Ho_{i}'].T
                dC_t = dC * Af 

                dH_next = dH_t

        return

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-x))
