import numpy as np
import pandas as pd
from typing import Optional
from scipy.optimize import minimize

class SVM:
    """SVM classifier from scratch."""
    def __init__(self, kernel: str='linear', learning_rate: float=0.1, C: float=1.0, polynomial_degree: Optional[int]=3, n_iter: int=1000, tol: float=1e-3):
        self.kernel = kernel
        self.lr = learning_rate
        self.C = C
        self.polynomial_degree = polynomial_degree
        self.n_iter = n_iter
        self.tol = tol
        
        self.w = None
        self.b = None
        self.alpha = None 
        self.dict_sv = {'alpha': None, 'X': None, 'y': None}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'SVM':
        """Train the SVM classifier."""
        y = np.where(y == 1, 1, -1)

        if self.kernel == 'linear':
            self.soft_margin_classifier(y, X)
        else:
            self.dual_problem(y, X)

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class labels for the input data."""
        if self.alpha is not None:
            return self._predict_dual(X)
        elif self.w is not None:
            return np.where(X @ self.w + self.b >= 0, 1, -1)
        else:
            raise ValueError("Model is not trained yet.")

    def _predict_dual(self, X: np.ndarray) -> np.ndarray:  
        """Predict class labels using the dual formulation."""
        K = self._kernel_matrix_predict(X)
        decision = np.sum(self.dict_sv['alpha'] * self.dict_sv['y'] * K, axis=1) + self.b
        return np.where(decision >= 0, 1, -1)

    def hard_margin_classifier(self, y: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Solve the hard margin optimization problem for SVM. IT assumes data is linearly separable."""
        self.w = np.zeros(X.shape[1])
        self.b = 0.0

        def loss_function(y: np.ndarray, X: np.ndarray, w: np.ndarray, b: float) -> float:
            """ Hard margin loss function minimizing ||w||^2 / 2"""
            return 0.5 * np.dot(w, w)

        def gradient_descent(y: np.ndarray, X: np.ndarray, w: np.ndarray, b: float) -> tuple[np.ndarray, float]:
            """Perform gradient descent to update weights and bias."""
            margins = y * (X @ w + b)
            grad_w = w - 1/X.shape[0] * np.sum(y[margins < 1, None] * X[margins < 1], axis=0)
            grad_b = b - 1/X.shape[0] * np.sum(y[margins < 1])
            self.w = w - self.lr * grad_w
            self.b = b - self.lr * grad_b
            return self.w, self.b

        list_loss = [np.inf]
        for n in range(self.n_iter):
            self.w, self.b = gradient_descent(y, X, self.w, self.b)
            loss =  loss_function(y, X, self.w, self.b)
            list_loss.append(loss)
            if np.abs(list_loss[-1] - list_loss[-2]) < self.tol:
                break
        
        return self

    def soft_margin_classifier(self, y: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Solve the soft margin optimization problem for SVM. It allows some misclassifications."""
        self.w = np.zeros(X.shape[1])
        self.b = 0.0

        def loss_function(y: np.ndarray, X: np.ndarray, w: np.ndarray, b: float) -> float:
            """ Soft margin loss function minimizing ||w||^2 / 2 + C * sum(xi) """
            margins = y * (X @ w + b)
            hinge_loss = np.maximum(0, 1 - margins)
            return 0.5 * np.dot(w, w) + self.C * np.mean(hinge_loss)

        def gradient_descent(y: np.ndarray, X: np.ndarray, w: np.ndarray, b: float) -> tuple[np.ndarray, float]:
            """Perform gradient descent to update weights and bias."""
            margins = y * (X @ w + b)

            grad_w = w - self.C / X.shape[0] * np.sum(y[margins < 1, None] * X[margins < 1], axis=0)
            grad_b = -1 * self.C / X.shape[0] * np.sum(y[margins < 1])

            self.w = w - self.lr * grad_w
            self.b = b - self.lr * grad_b
            return self.w, self.b

        list_loss = [np.inf]
        for n in range(self.n_iter):
            self.w, self.b = gradient_descent(y, X, self.w, self.b)
            loss =  loss_function(y, X, self.w, self.b)
            list_loss.append(loss)
            if np.abs(list_loss[-1] - list_loss[-2]) < self.tol:
                break

        return self

    def dual_problem(self, y: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Solve the dual optimization problem for SVM."""
        K = self._kernel_matrix(X)
        H = np.outer(y, y) * K

        def lagrange_objective(alpha: np.ndarray) -> np.ndarray:
            """Solve the dual optimization problem for SVM."""
            return 0.5 * alpha.T @ H @ alpha - np.sum(alpha)

        constraints = {'type': 'eq', 'fun': lambda alpha: np.sum(alpha * y)}
        alpha0 = np.zeros(X.shape[0])
        opt = minimize(lagrange_objective, alpha0, method='SLSQP', bounds=[(0, None)], constraints=constraints, options={'maxiter': self.n_iter, 'ftol': self.tol})
        self.alpha = opt.x

        idx = np.where(self.alpha > self.tol)[0]
        self.dict_sv = {'alpha': self.alpha[idx], 'X': X[idx], 'y': y[idx]}
        if len(idx) > 0:
            sv_K = self._kernel_matrix(self.dict_sv['X'])
            self.b = np.mean(self.dict_sv['y']) - np.sum(self.dict_sv['alpha'] * self.dict_sv['y'] * sv_K, axis=1)
            
        if self.kernel == 'linear':
            self.w = np.sum(self.dict_sv['alpha'] * self.dict_sv['y'][:, None] * self.dict_sv['X'], axis=0)

        return self

    def _kernel_matrix(self, X: np.ndarray) -> np.ndarray:
        """Compute the kernel matrix."""
        N = X.shape[0]
        K = np.zeros((N, N))

        if self.kernel == 'linear':
            K = X @ X.T
        elif self.kernel == 'polynomial':
            degree = self.polynomial_degree
            coef0 = 1
            K = (X @ X.T + coef0) ** degree
        elif self.kernel == 'rbf':
            gamma = 1.0 / X.shape[1]
            for i in range(N):
                for j in range(N):
                    diff = X[i] - X[j]
                    K[i, j] = np.exp(-gamma * np.dot(diff, diff))
        else:
            raise ValueError("Unsupported kernel type")

        return K

    def _kernel_matrix_predict(self, X: np.ndarray) -> np.ndarray:
        """Compute the kernel matrix between support vectors and input data for prediction."""
        N_sv = self.dict_sv['X'].shape[0]
        N = X.shape[0]
        K = np.zeros((N, N_sv))

        if self.kernel == 'linear':
            K = X @ self.dict_sv['X'].T
        elif self.kernel == 'polynomial':
            degree = self.polynomial_degree
            coef0 = 1
            K = (X @ self.dict_sv['X'].T + coef0) ** degree
        elif self.kernel == 'rbf':
            gamma = 1.0 / X.shape[1]
            for i in range(N):
                for j in range(N_sv):
                    diff = X[i] - self.dict_sv['X'][j]
                    K[i, j] = np.exp(-gamma * np.dot(diff, diff))
        else:
            raise ValueError("Unsupported kernel type")

        return K