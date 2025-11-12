from typing import List, Optional

import numpy as np
from .base import BaseLinearRegression;

class LinearRegressionGD(BaseLinearRegression):
    """
    Linear Regression using Gradient Descent with SSE cost function.

    Parameters
    ----------
    learning_rate : float, default=0.01
        The step size for gradient descent updates
    max_iter : int, default=1000
        Maximum number of iterations for gradient descent
    tol : float, default=1e-6
        Tolerance for convergence criterion
    batch_size : int, default=32
        Size of mini-batches for gradient descent
    random_state : int, optional
        Random seed for reproducibility
    fit_intercept : bool, default=True
        Whether to fit an intercept term
    """

    def __init__(self,
                 learning_rate: float = 0.01,
                 max_iter: int = 1000,
                 tol: float = 1e-6,
                 batch_size: int = 32,
                 random_state: Optional[int] = None,
                 fit_intercept: bool = True):
        super().__init__(fit_intercept=fit_intercept)
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.batch_size = batch_size
        self.random_state = random_state
        self.cost_history_: List[float] = []
        self.n_iter_: Optional[int] = None

    def _compute_cost(self, X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> float:
        """
        Compute the SSE cost function.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix with intercept column if fit_intercept=True
        y : np.ndarray
            Target values
        theta : np.ndarray
            Parameter vector

        Returns
        -------
        cost : float
            SSE cost value
        """
        # J(theta) = (1/2m) * sum((X @ theta - y)^2)
        m = X.shape[0]  # number of samples
        predictions = X @ theta
        cost = (1/(2 * m)) * np.sum((predictions - y.reshape(-1,1))**2)
        return cost

    def _compute_gradients(self, X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """
        Compute gradients of the cost function.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix with intercept column if fit_intercept=True
        y : np.ndarray
            Target values
        theta : np.ndarray
            Parameter vector

        Returns
        -------
        gradients : np.ndarray
            Gradient vector
        """
        m = X.shape[0]
        predictions = X @ theta
        error = predictions - y.reshape(-1, 1)
        # partial deriviative via matrix
        gradients = (1/m) * X.T @ error
        return gradients

    def _get_mini_batches(self, X: np.ndarray, y: np.ndarray) -> list:
        """Generate mini-batches for stochastic gradient descent."""
        
        # Create mini-batches
        pass

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearRegressionGD':
        """
        Fit the model using gradient descent.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values

        Returns
        -------
        self : LinearRegressionGD
            Returns self for method chaining
        """
        np.random.seed(self.random_state)

        # Validate and preprocess input
        X, y = self._validate_input(X, y)
        
        # Add intercept column if needed
        X = self._add_intercept(X)
        
        # Initialize theta to zeros 
        theta = np.zeros((X.shape[1], 1))
            
        for i in range(self.max_iter):
            # current cost
            current_cost = self._compute_cost(X, y, theta)
            self.cost_history_.append(current_cost)

            # which way do I move
            gradients = self._compute_gradients(X, y, theta)
            
            # update thetas
            theta = theta - self.learning_rate * gradients
            
            # Check for NaN
            if np.isnan(theta).any():
                print(f"NaN detected at iteration {i}")
                break
            
            if len(self.cost_history_) > 1:
                cost_change = abs(self.cost_history_[-2] - current_cost)
                if cost_change < self.tol:
                    break
                
        if self.fit_intercept:
            self.intercept_ = theta[0, 0]  
            self.coef_ = theta[1:, :]     
        else:
            self.coef_ = theta
        
        self.n_iter_ = i + 1

        return self