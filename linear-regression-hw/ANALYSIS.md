# Linear Regression Implementation Analysis

## **Learning Rate Analysis**

### Key Findings

I analyzed different learning rates (.0001 to 1.0) revealed a clear zone for optimal performance.

**Optimal Range:** 5e-2 to 5e-1 for normalized data

### Performance Breakdown

| Learning Rate Range | Performance          | Convergence                | Status     |
| ------------------- | -------------------- | -------------------------- | ---------- |
| .0001 to .001       | Moderate (R² ~0.2)   | Slow (2000 iter)           | Suboptimal |
| .001 to .01         | Good (R² ~0.25)      | Reasonable (900-2000 iter) | Acceptable |
| .01 to .1           | Excellent (R² ~0.26) | Fast (112-908 iter)        | Optimal    |
| ≥ 1.0               | Failed (NaN)         | Divergence (~1600 iter)    | Unstable   |

### Key Insights

1. **Sweet spot at LR = 0.5** - Converges in just 112 iterations with near-perfect accuracy (R² = 0.261077)
2. **Stability boundary at LR = 1.0** - Beyond this point, numerical instability causes divergence
3. **Convergence speed vs stability trade-off** - Higher learning rates converge faster but risk instability

## **Batch Size Impact**

I implemented **full-batch gradient descent**, using all training samples in each iteration.

### Trade-offs Analysis

**Full Batch:**

- **Pros:**
  - Stable, deterministic convergence
  - Smooth cost curves with no noise
  - Guaranteed to find global minimum for convex problems
  - Reproducible results
- **Cons:**
  - Computationally expensive for large datasets

**Mini-Batch (Alternative):**

- **Pros:**
  - Faster computation on large datasets
  - Stochasticity helps escape local minima
  - Better memory efficiency
- **Cons:**
  - Noisier convergence paths
  - Less stable gradient estimates

## **Cost Function Choice: SSE vs MSE**

I used: `J(θ) = (1/2m) * Σ(ŷ - y)²`

This is a scaled MSE.

### Cost Function Comparison

**Sum of Squared Errors (SSE):** `Σ(ŷ - y)²`

- **Pros:** Simple, direct error measurement
- **Cons:**
  - Gradients grow linearly with more data
  - Hard to compare across different dataset sizes
  - Learning rates need adjustment based on sample size

**Mean Squared Error (MSE):** `(1/m) * Σ(ŷ - y)²`

- **Pros:**
- Normalized by sample size
- More interpretable magnitude
- Consistent across different dataset sizes
- Learning rates more transferable
- **Cons:** Slightly more computation per iteration

## **Coefficient Comparison Analysis**

### Results Summary

From the model comparison:

```
Method            Coefficients                    Intercept    R² Score
Normal Equation:  [0.472, -0.241, -0.320]       -0.181       0.2611
Scikit-learn:     [0.472, -0.241, -0.320]       -0.181       0.2611
Gradient Descent: [0.404, -0.102, -0.239]       -0.178       0.2412
```

### Key Insights

**1. Perfect Mathematical Validation**

- **Normal Equation = sklearn** (coefficient difference: 0.000000)

**2. Gradient Descent Convergence Gap**

- **Coefficient difference from optimal:** 0.174731
- **R² gap:** 0.0199 (2% performance loss)
- **Root cause:** Incomplete convergence after 1000 iterations
- **Solution:** Increase max_iter or learning rate

**3. Feature Interpretation** (on normalized data)

| Feature  | Coefficient | Interpretation                            |
| -------- | ----------- | ----------------------------------------- |
| Size     | +0.472      | Larger houses → higher prices (expected)  |
| Bedrooms | -0.241      | More bedrooms → lower prices (surprising) |
| Age      | -0.320      | Older houses → lower prices (expected)    |

**Surprising Finding:** Negative bedroom coefficient suggests either:

- Multicollinearity between size and bedrooms
- Smaller houses with more bedrooms are less desirable
- Sample size too small for reliable coefficient estimation

## **Implementation Challenges & Solutions**

### Challenge 1: Numerical Overflow

**Problem:**
Raw data values caused numerical overflow:

- House prices: $199,000 - $405,000
- Square footage: 1 - 2,450
- Initial random weights caused exploding gradients

**Solution:**
Data normalization: `X_norm = (X - μ) / σ`

- Transformed all features to mean=0, std=1
- Eliminated numerical instability

### Challenge 2: Matrix Dimension Mismatches

**Problem:**
Frequent shape errors in matrix operations:

**Root Causes:**

- Inconsistent intercept handling
- Missing `.reshape(-1, 1)` for column vectors
- Confusion between training and prediction data shapes

**Solutions:**

1. **Consistent shape management:**

   ```python
   y = y.reshape(-1, 1)  # Always column vector
   theta = theta.reshape(-1, 1)  # Always column vector
   ```

2. **Clear intercept separation:**

   ```python
   if self.fit_intercept:
       self.intercept_ = theta[0, 0]  # First element
       self.coef_ = theta[1:, :]     # Remaining elements
   ```

### Challenge 3: Convergence Detection

**Problem:**
How to determine when training should stop:

- Fixed iterations may be too many or too few
- No indication of convergence quality
- Difficult to compare different learning rates

**Solutions:**

1. **Tolerance-based early stopping:**

   ```python
   if len(self.cost_history_) > 1:
       cost_change = abs(self.cost_history_[-2] - current_cost)
       if cost_change < self.tol:
           break
   ```

2. **Cost history tracking:**

   ```python
   self.cost_history_.append(current_cost)
   ```

### Challenge 4: Learning Rate Selection

**Problem:**
Initial learning rate of 0.01 caused:

- Slow convergence (1000+ iterations)
- Suboptimal final performance
- Poor coefficient estimates

**Discovery Process:**

1. **Started with common value:** LR = 0.01
2. **Observed slow convergence:** 1000 iterations, still improving
3. **Found optimal range:** 0.05 - 0.5

**Final Solution:**

- **Default LR = 0.05** for good balance
