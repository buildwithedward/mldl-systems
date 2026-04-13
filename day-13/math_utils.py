# maths_utils.py — save this file
import numpy as np

def normalise(x: np.ndarray) -> np.ndarray:
    """Normalise array to zero mean and unit variance."""
    return (x - x.mean()) / x.std()

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error between two arrays."""
    return float(np.sqrt(((y_true - y_pred) ** 2).mean()))