# Polynomial Regression

Polynomial regression models a **non-linear relationship** between `x` and `y` by extending linear regression with polynomial features.

## Why Use It?

Linear regression assumes `y` changes in a straight line with `x`. When the data is curved, a straight line underfits. Polynomial regression fixes this by adding higher-power terms of the input as new columns, then fitting an ordinary linear model on them.

The model is still *linear in its coefficients*, which is why it is called polynomial **linear** regression.

## How It Works

1. Take the input column(s).
2. Generate polynomial features up to the chosen **degree**.
3. Train a standard linear regression model on the expanded features.

### Example (degree = 2)

For a single input `x`, the feature columns become:

| 1 (bias) | x | x² |
|----------|---|----|

The model learns:

```
y = b0 + b1·x + b2·x²
```

For two inputs `x1` and `x2` with degree 2, the features are:

```
1, x1, x2, x1², x1·x2, x2²
```

## Code

```python
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

# sample non-linear data
X = np.linspace(-3, 3, 100).reshape(-1, 1)
y = 0.5 * X.ravel() ** 2 + X.ravel() + np.random.randn(100) * 0.5

# degree=2 -> features: 1, x, x^2
model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
model.fit(X, y)

y_pred = model.predict(X)
```

## Choosing the Degree

- **Too low** (e.g. 1): underfitting, the curve misses the pattern.
- **Too high**: overfitting, the curve chases noise and generalizes poorly.
- Pick the degree using **cross-validation** or validation error, and consider regularization (Ridge/Lasso) for higher degrees.

## Key Points

- Use when the relationship between `x` and `y` is curved.
- Feature scaling helps for higher degrees.
- Number of features grows quickly with degree and number of inputs.
