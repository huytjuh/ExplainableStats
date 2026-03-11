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

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'LSTM':
        """Train the LSTM classifier."""
        X = X.values if isinstance(X, pd.DataFrame) else X
        y = y.values if isinstance(y, (pd.Series, pd.DataFrame)) else y
        n_samples, timesteps, n_features = X.shape

        self._initialize_weights(n_features)
        list_loss = [np.inf]

        self._forward_propagation(X, n_samples, timesteps)

        # y_pred = self._forward_propagation(X, n_samples, timesteps)
        
        # self._backward_propagation(X, y, y_pred, timesteps)

        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class labels for the input data."""
        return np.zeros(X.shape[0])
    
    def _initialize_weights(self, n_features: int) -> None:
        """Initialize weights and biases for the network."""
        self.W, self.b = {}, {}
        prev_layer_size = n_features
        for i in range(len(self.hidden_layers)):
            h_size = self.hidden_layers[i]
            scale_x = np.sqrt(2 / prev_layer_size)
            for gate in ['f', 'i', 'o', 'g']:
                self.W[f'W{gate}_{i}'] = np.random.randn(prev_layer_size, h_size) * scale_x
                self.b[f'b{gate}_{i}'] = np.zeros((1, h_size))
            
            prev_layer_size = h_size

    def _forward_propagation(self, X: np.ndarray, n_samples: int, timesteps: int) -> np.ndarray:
        """Perform forward propagation through the network."""
        
        H = [np.zeros((n_samples, timesteps, h_size)) for h_size in self.hidden_layers]
        C = [np.zeros((n_samples, timesteps, h_size)) for h_size in self.hidden_layers]

        for i in range(len(self.hidden_layers)):
            for t in range(timesteps):
                X_t = X[:, t, :] if i==0 else H[i-1][:, t, :]
                C_t = C[i][:, t-1, :] if t>0 else C[i][:, 0, :]
                H_t = H[i][:, t-1, :] if t>0 else H[i][:, 0, :]

                XH = np.concatenate([X_t, H_t], axis=1)

                print(X_t.shape, H_t.shape, XH.shape)
                
                # PREACTIVATION GATES
                Zf = XH @ self.W[f'Wf_{i}'] + self.b[f'bf_{i}']
                Zi = XH @ self.W[f'Wi_{i}'] + self.b[f'bi_{i}']
                Zg = XH @ self.W[f'Wg_{i}'] + self.b[f'bg_{i}']
                Zo = XH @ self.W[f'Wo_{i}'] + self.b[f'bo_{i}']

                # ACTIVATION GATES
                Af = self._sigmoid(Zf)
                Ai = self._sigmoid(Zi)
                Ag = np.tanh(Zg)
                Ao = self._sigmoid(Zo)

                print(X_t)


                Zg = [X_t, H_t] @ self.W[f'Wg_{i}'] + self.b[f'bg_{i}']
                Zo = [X_t, H_t] @ self.W[f'Wo_{i}'] + self.b[f'bo_{i}']


        # self.W, self.U, self.b = {}, {}, {}
        # prev_layer_size = n_features
        # for i in range(len(self.hidden_layers)):
        #     h_size = self.hidden_layers[i]
        #     scale_x = np.sqrt(2 / prev_layer_size)
        #     scale_h = np.sqrt(2 / h_size)
        #     for gate in ['f', 'i', 'o', 'g']:
        #         self.W[f'W{gate}_{i}'] = np.random.randn(prev_layer_size, h_size) * scale_x
        #         self.U[f'U{gate}_{i}'] = np.random.randn(h_size, h_size) * scale_h
        #         self.b[f'b{gate}_{i}'] = np.zeros((1, h_size))

        #     prev_layer_size = h_size

        # scale_x = np.sqrt(2 / prev_layer_size)
        # self.W['W_out'] = np.random.randn(self.hidden_layers[-1], 1) * scale_x
        # self.b['b_out'] = np.zeros((1, 1))

    # def _forward_propagation(self, X: np.ndarray, n_samples: int, timesteps: int) -> None:
    #     """Perform forward propagation through the network."""

    #     for i in range(len(self.hidden_layers)):
    #         self.cache[f'C_{i}_0'] = np.zeros((n_samples, self.hidden_layers[i]))
    #         self.cache[f'H_{i}_0'] = np.zeros((n_samples, self.hidden_layers[i]))

    #     for t in range(timesteps):
    #         X_t = X[:, t, :]

    #         for i in range(len(self.hidden_layers)):
    #             C_prev  = self.cache[f'C_{i}_{t}']
    #             H_prev  = self.cache[f'H_{i}_{t}']

    #             # PRE ACTIVATION GATES
    #             Z_f = X_t @ self.W[f'W_f_{i}'] + H_prev @ self.U[f'U_f_{i}'] + self.b[f'b_f_{i}']
    #             Z_i = X_t @ self.W[f'W_i_{i}'] + H_prev @ self.U[f'U_i_{i}'] + self.b[f'b_i_{i}']
    #             Z_g = X_t @ self.W[f'W_g_{i}'] + H_prev @ self.U[f'U_g_{i}'] + self.b[f'b_g_{i}']
    #             Z_o = X_t @ self.W[f'W_o_{i}'] + H_prev @ self.U[f'U_o_{i}'] + self.b[f'b_o_{i}']

    #             # ACTIVATION GATES
    #             A_f = self._sigmoid(Z_f)
    #             A_i = self._sigmoid(Z_i)
    #             A_g = np.tanh(Z_g)
    #             A_o = self._sigmoid(Z_o)

    #             # CELL STATE
    #             C = A_f * C_prev + A_i * A_g
    #             H = A_o * np.tanh(C)

    #             # CACHE
    #             for keys in ['Z_f', 'Z_i', 'Z_g', 'Z_o', 'A_f', 'A_i', 'A_g', 'A_o', 'C', 'H']:
    #                 self.cache[f'{keys}_{i}_{t+1}'] = eval(keys)

    #             H_prev = H

    #     # FINAL OUTPUT
    #     H_out = H_prev 
    #     Z_out = H_out @ self.W['W_out'] + self.b['b_out']
    #     A_out = self._sigmoid(Z_out)

    #     # CACHE FINAL OUTPUT
    #     for keys in ['H_out', 'Z_out', 'A_out']:
    #         self.cache[keys] = eval(keys)

    #     return A_out


    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-x))

    def _sigmoid_derivative(self, x: np.ndarray) -> np.ndarray:
        """Derivative of sigmoid function."""
        return self._sigmoid(x) * (1 - self._sigmoid(x))
    


                




        for t in range(timesteps):
            for i in range(len(self.hidden_layers)):
                h_prev = self.cache.get(f'H_{i}', np.zeros((n_samples, self.hidden_layers[i])))
                x_t = self.cache['A_0'][:, t, :]
                gates = {}
                for gate in ['f', 'i', 'o', 'g']:
                    Z = x_t @ self.W[f'W_{gate}_{i}'] + h_prev @ self.U[f'U_{gate}_{i}'] + self.b[f'b_{gate}_{i}']
                    gates[gate] = Z

                f_t = self._activation_func(gates['f'], method='sigmoid')
                i_t = self._activation_func(gates['i'], method='sigmoid')
                o_t = self._activation_func(gates['o'], method='sigmoid')
                g_t = self._activation_func(gates['g'], method='tanh')

                c_t = f_t * h_prev + i_t * g_t
                h_t = o_t * np.tanh(c_t)

                self.cache[f'H_{i}'] = h_t

        Z_out = self.cache[f'H_{len(self.hidden_layers)-1}'] @ self.W['W_out'] + self.b['b_out']