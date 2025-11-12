import pandas as pd
import numpy as np


def load_data(path='data/housing_data.csv'):
    df = pd.read_csv(path)
    X = df[['size', 'bedrooms', 'age']].values
    y = df['price'].values
    return X,y

def split_data(X, y, test_size=0.2, random_state = 1):
    np.random.seed(random_state)

    n_samples = len(X)
    n_test = int(n_samples * test_size)
    
    # Shuffle indices
    indices = np.random.permutation(n_samples)
    
    # Split indices
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    # Split data
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    from lr.gradual_descent import LinearRegressionGD
    from lr.normal_equation import LinearRegressionNE
    from sklearn.linear_model import LinearRegression as SklearnLinearRegression
    import time
    import matplotlib.pyplot as plt
    
    # Load and split data  
    X, y = load_data()
    
    # Normalize data to prevent numerical issues
    X_mean, X_std = X.mean(axis=0), X.std(axis=0)
    y_mean, y_std = y.mean(), y.std()
    
    X = (X - X_mean) / X_std
    y = (y - y_mean) / y_std
    
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    print("=" * 60)
    print("LINEAR REGRESSION IMPLEMENTATION COMPARISON")
    print("=" * 60)
    
    # Train Gradient Descent model
    print("\n1. GRADIENT DESCENT:")
    gd_model = LinearRegressionGD(learning_rate=0.01, max_iter=1000)
    gd_model.fit(X_train, y_train)
    
    gd_train_score = gd_model.score(X_train, y_train)
    gd_test_score = gd_model.score(X_test, y_test)

    print(f"   Training R²: {gd_train_score:.4f}")
    print(f"   Test R²: {gd_test_score:.4f}")
    print(f"   Converged in {gd_model.n_iter_} iterations")
    print(f"   Coefficients: {gd_model.coef_.flatten()}")
    print(f"   Intercept: {gd_model.intercept_:.4f}")
    
    # Train Normal Equation model
    print("\n2. NORMAL EQUATION:")
    ne_model = LinearRegressionNE()
    ne_model.fit(X_train, y_train)
    
    ne_train_score = ne_model.score(X_train, y_train)
    ne_test_score = ne_model.score(X_test, y_test)
    
    print(f"   Training R²: {ne_train_score:.4f}")
    print(f"   Test R²: {ne_test_score:.4f}")
    print(f"   Coefficients: {ne_model.coef_.flatten()}")
    print(f"   Intercept: {ne_model.intercept_:.4f}")
    
    print("\n3. SCIKIT-LEARN (REFERENCE):")
    sk_model = SklearnLinearRegression()
    sk_model.fit(X_train, y_train)
    
    sk_train_score = sk_model.score(X_train, y_train)
    sk_test_score = sk_model.score(X_test, y_test)
    
    print(f"   Training R²: {sk_train_score:.4f}")
    print(f"   Test R²: {sk_test_score:.4f}")
    print(f"   Coefficients: {sk_model.coef_}")
    print(f"   Intercept: {sk_model.intercept_:.4f}")
    
    # Compare all results
    print("\n4. COMPARISON:")    
    print(f"\n   Test R² Scores:")
    print(f"     Gradient Descent: {gd_test_score:.4f}")
    print(f"     Normal Equation:  {ne_test_score:.4f}")
    print(f"     Scikit-learn:     {sk_test_score:.4f}")
    
    print(f"\n   Accuracy vs Scikit-learn:")
    print(f"     Normal Equation R² diff: {abs(ne_test_score - sk_test_score):.6f}")
    print(f"     Normal Equation coef diff: {np.linalg.norm(ne_model.coef_.flatten() - sk_model.coef_):.6f}")
    print(f"     Gradient Descent R² diff: {abs(gd_test_score - sk_test_score):.6f}")
    print(f"     Gradient Descent coef diff: {np.linalg.norm(gd_model.coef_.flatten() - sk_model.coef_):.6f}")

    
    # Generate predictions for visualization
    gd_pred = gd_model.predict(X_test)
    ne_pred = ne_model.predict(X_test)
    sk_pred = sk_model.predict(X_test)
    
    # Create visualizations
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS...")
    print("="*60)
    
    # Set up the plot style
    plt.style.use('default')
    fig = plt.figure(figsize=(15, 10))
    
    # 1. Actual vs Predicted scatter plots
    fig.suptitle('Linear Regression Model Comparison', fontsize=16, fontweight='bold')
    
    # Plot 1: Gradient Descent
    ax1 = plt.subplot(2, 3, 1)
    plt.scatter(y_test, gd_pred, alpha=0.7, color='red', s=60)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Gradient Descent\nR² = {gd_test_score:.4f}')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Normal Equation
    ax2 = plt.subplot(2, 3, 2)
    plt.scatter(y_test, ne_pred, alpha=0.7, color='blue', s=60)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Normal Equation\nR² = {ne_test_score:.4f}')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Scikit-learn
    ax3 = plt.subplot(2, 3, 3)
    plt.scatter(y_test, sk_pred, alpha=0.7, color='green', s=60)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Scikit-learn\nR² = {sk_test_score:.4f}')
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Cost history for Gradient Descent
    ax4 = plt.subplot(2, 3, (4, 6))  # Span bottom row
    iterations = range(1, len(gd_model.cost_history_) + 1)
    plt.plot(iterations, gd_model.cost_history_, 'r-', linewidth=2, label='Training Cost')
    plt.xlabel('Iteration')
    plt.ylabel('Cost (MSE)')
    plt.title('Gradient Descent Cost History')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add convergence info
    final_cost = gd_model.cost_history_[-1]
    plt.text(0.02, 0.98, f'Final Cost: {final_cost:.6f}\nIterations: {gd_model.n_iter_}', 
             transform=ax4.transAxes, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot instead of showing interactively
    plt.savefig('outputs/linear_regression_comparison.png', dpi=150, bbox_inches='tight')
    print("   Plot saved to outputs/linear_regression_comparison.png")
    plt.close()
    
    # Print summary statistics
    print("\n5. PREDICTION QUALITY:")
    
    # Calculate residuals
    gd_residuals = y_test - gd_pred
    ne_residuals = y_test - ne_pred  
    sk_residuals = y_test - sk_pred
    
    print(f"   Mean Absolute Error:")
    print(f"     Gradient Descent: {np.mean(np.abs(gd_residuals)):.4f}")
    print(f"     Normal Equation:  {np.mean(np.abs(ne_residuals)):.4f}")
    print(f"     Scikit-learn:     {np.mean(np.abs(sk_residuals)):.4f}")
    
    print(f"\n   Root Mean Square Error:")
    print(f"     Gradient Descent: {np.sqrt(np.mean(gd_residuals**2)):.4f}")
    print(f"     Normal Equation:  {np.sqrt(np.mean(ne_residuals**2)):.4f}")
    print(f"     Scikit-learn:     {np.sqrt(np.mean(sk_residuals**2)):.4f}")
    
    