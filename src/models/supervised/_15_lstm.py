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
            self.W[i], self.H[i], self.b[i] = {}, {}, {}
            h_size = self.hidden_layers[i]
            scale_x = np.sqrt(2 / prev_layer_size)
            for gate in ['f', 'i', 'c', 'o']:
                self.W[i][f'W{gate}'] = np.random.randn(prev_layer_size, h_size) * scale_x
                self.H[i][f'H{gate}'] = np.random.randn(h_size, h_size) * scale_x
                self.b[i][f'b{gate}'] = np.zeros((1, h_size))
            
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
                Zf = X_t @ self.W[i][f'Wf'] + H_prev @ self.H[i][f'Hf'] + self.b[i][f'bf']
                Zi = X_t @ self.W[i][f'Wi'] + H_prev @ self.H[i][f'Hi'] + self.b[i][f'bi']
                Zc = X_t @ self.W[i][f'Wc'] + H_prev @ self.H[i][f'Hc'] + self.b[i][f'bc']
                Zo = X_t @ self.W[i][f'Wo'] + H_prev @ self.H[i][f'Ho'] + self.b[i][f'bo']

                # ACTIVATION GATES
                Af = self._sigmoid(Zf)      # FORGET GATE 
                Ai = self._sigmoid(Zi)      # INPUT GATE 
                Ac = np.tanh(Zc)            # INPUT GATE
                Ao = self._sigmoid(Zo)      # OUTPUT GATE

                # CELL STATE AND HIDDEN STATE
                C[i][:, t, :] = C_prev * Af + Ai * Ac
                H[i][:, t, :] = Ao * np.tanh(C[i][:, t, :])

                for key in ['Zf', 'Zi', 'Zc', 'Zo', 'Af', 'Ai', 'Ac', 'Ao']:
                    self.cache[f'{key}_{i}_{t}'] = eval(key)

        # OUTPUT LAYER
        H_out = H[-1][:, -1, :]
        Z_out = H_out @ self.W['W_out'] + self.b['b_out']
        A_out = self._sigmoid(Z_out).ravel()

        # CACHE
        for keys in ['C', 'H', 'Z_out', 'A_out']:
            self.cache[f'{keys}'] = eval(keys)

        return A_out
    
    def _backward_propagation(self, X: np.ndarray, y: np.ndarray, y_pred: np.ndarray) -> None:
        """Perform backward propagation and update weights."""
        n_samples, timesteps, n_features = X.shape

        y_pred = np.clip(y_pred, 1e-9, 1 - 1e-9)
        dZ_out = y_pred.reshape(-1, 1) - y                                 # BINARY CROSS-ENTROPY LOSS DERIVATIVE
        dW_out = self.cache['H'][-1][:, -1, :].T @ dZ_out / n_samples      # dL/dW = dL/dZ * dZ/dW_ = H.T @ dZ
        db_out = np.sum(dZ_out, axis=0, keepdims=True) / n_samples         # dL/db = dL/dZ * dZ/db = sum(dZ) * 1
        self.W['W_out'] -= self.lr * dW_out
        self.b['b_out'] -= self.lr * db_out

        dH_out = dZ_out @ self.W['W_out'].T / n_samples                    # dL/dH = dL/dZ * dZ/dH = dZ @ W.T
        dH_into_layer = dH_out
        for i in range(len(self.hidden_layers)-1, -1, -1):
            dH_next, dH_into_layer = dH_into_layer, None

            dH_t = np.zeros((n_samples, self.hidden_layers[i]))
            dC_t = np.zeros((n_samples, self.hidden_layers[i]))
           
            for t in range(timesteps-1, -1, -1):
                # GET FIRST STEP OF EACH SAMPLE
                X_t = X[:, t, :] if i==0 else self.cache['H'][i-1][:, t, :]

                # GET PREVIOUS CELL STATE AND HIDDEN STATE
                C_prev = self.cache['C'][i][:, t-1, :] if t>0 else np.zeros((n_samples, self.hidden_layers[i]))
                H_prev = self.cache['H'][i][:, t-1, :] if t>0 else np.zeros((n_samples, self.hidden_layers[i]))

                # GATE VALUES
                Af = self.cache[f'Af_{i}_{t}']
                Ai = self.cache[f'Ai_{i}_{t}']  
                Ac = self.cache[f'Ac_{i}_{t}']
                Ao = self.cache[f'Ao_{i}_{t}']
                C_t_val = self.cache['C'][i][:, t, :]

                dH_total = (dH_next if t == timesteps-1 else np.zeros_like(dH_t)) + dH_t

                # ── CELL STATE GRADIENT (two paths) ───────────────────────────────
                # Path 1: H_t = Ao * tanh(C_t)
                # Path 2: C_{t+1} = Af_{t+1} * C_t  (carried via dC_t)
                dC = dH_total * Ao * (1 - np.tanh(C_t_val)**2) + dC_t

                # DERIVATIVE OF ACTIVATION GATES
                dAf = dC * C_prev
                dAi = dC * Ac
                dAc = dC * Ai
                dAo = dH_total * np.tanh(C_t_val)

                # DERIVATIVE OF PREACTIVATION GATES
                dZf = dAf * Af * (1 - Af)       # sigmoid'(Zf) = Af * (1 - Af)
                dZi = dAi * Ai * (1 - Ai)       # sigmoid'(Zi) = Ai * (1 - Ai)
                dZc = dAc * (1 - Ac**2)         # tanh'(Zc) = 1 - tanh(Zc)^2 = 1 - Ac^2
                dZo = dAo * Ao * (1 - Ao)       # sigmoid'(Zo) = Ao * (1 - Ao)

                # GRADIENTS FOR WEIGHTS AND BIASES
                dWf = X_t.T @ dZf / n_samples
                dWi = X_t.T @ dZi / n_samples
                dWc = X_t.T @ dZc / n_samples
                dWo = X_t.T @ dZo / n_samples
                dbf = np.sum(dZf, axis=0, keepdims=True) / n_samples
                dbi = np.sum(dZi, axis=0, keepdims=True) / n_samples
                dbc = np.sum(dZc, axis=0, keepdims=True) / n_samples
                dbo = np.sum(dZo, axis=0, keepdims=True) / n_samples
                dHf = H_prev.T @ dZf / n_samples
                dHi = H_prev.T @ dZi / n_samples
                dHc = H_prev.T @ dZc / n_samples
                dHo = H_prev.T @ dZo / n_samples

                # UPDATE WEIGHTS AND BIASES
                self.W[i][f'Wf'] -= self.lr * dWf
                self.W[i][f'Wi'] -= self.lr * dWi  
                self.W[i][f'Wc'] -= self.lr * dWc
                self.W[i][f'Wo'] -= self.lr * dWo
                self.b[i][f'bf'] -= self.lr * dbf
                self.b[i][f'bi'] -= self.lr * dbi
                self.b[i][f'bc'] -= self.lr * dbc
                self.b[i][f'bo'] -= self.lr * dbo
                self.H[i][f'Hf'] -= self.lr * dHf
                self.H[i][f'Hi'] -= self.lr * dHi
                self.H[i][f'Hc'] -= self.lr * dHc
                self.H[i][f'Ho'] -= self.lr * dHo

                # GRADIENT FOR PREVIOUS HIDDEN STATE (carried to next time step)
                dH_t = dZf @ self.H[i][f'Hf'].T + dZi @ self.H[i][f'Hi'].T + dZc @ self.H[i][f'Hc'].T + dZo @ self.H[i][f'Ho'].T
                dC_t = dC * Af 

                dH_into_layer = dZf @ self.W[i][f'Wf'].T + dZi @ self.W[i][f'Wi'].T + dZc @ self.W[i][f'Wc'].T + dZo @ self.W[i][f'Wo'].T

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-x))
