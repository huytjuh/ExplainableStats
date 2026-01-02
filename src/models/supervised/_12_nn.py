import pandas as pd 
import numpy as np 

class NeuralNetwork:
    """Neural Network classifier from scratch."""
    
    def __init__(self):
        pass

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'NeuralNetwork':
        """Train the Neural Network classifier."""
        pass

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class labels for the input data."""
        pass

    def _forward_propagation(self, X: pd.DataFrame) -> np.ndarray:
        """Perform forward propagation."""
        pass

    def _backward_propagation(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Perform backward propagation and update weights."""
        pass

    def _activation_func(self, y: np.ndarray, method='relu') -> np.ndarray:
        """ReLU activation function."""
        if method == 'relu':
            return np.maximum(0, y) 
        return
