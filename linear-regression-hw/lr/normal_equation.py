from typing import Optional
import numpy as np
from .base import BaseLinearRegression

class LinearRegressionNE(BaseLinearRegression):
    """
    Linear Regression using Normal Equation (closed-form solution).
    
    The normal equation provides the optimal solution directly:
    θ = (X^T X)^(-1) X^T y
    
    This is faster than gradient descent for small to medium datasets
    but becomes computationally expensive for large feature sets.
    
    Parameters
    ----------
    fit_intercept : bool, default=True
        Whether to fit an intercept term
    """
    
    def __init__(self, fit_intercept: bool = True):
        super().__init__(fit_intercept=fit_intercept)
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearRegressionNE':
        """
        Fit the model using the normal equation.
        
        The normal equation solves for optimal parameters in one step:
        θ = (X^T X)^(-1) X^T y
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values
            
        Returns
        -------
        self : LinearRegressionNE
            Returns self for method chaining
        """
        X, y = self._validate_input(X, y)
        
        X = self._add_intercept(X)
        
        # Reshape y to column vector for consistency
        y = y.reshape(-1, 1)
        
        # Solve normal equation: θ = (X^T X)^(-1) X^T y
        X_transpose = X.T
        XTX = X_transpose @ X
        XTy = X_transpose @ y
        
        try:
            theta = np.linalg.solve(XTX, XTy)
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse if matrix is singular
            theta = np.linalg.pinv(XTX) @ XTy
        
        # Store parameters
        if self.fit_intercept:
            self.intercept_ = theta[0, 0]  # First element is bias
            self.coef_ = theta[1:, :]     # Rest are feature weights
        else:
            self.intercept_ = 0.0
            self.coef_ = theta
            
        return self
