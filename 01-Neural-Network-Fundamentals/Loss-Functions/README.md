# Loss Functions

A **Loss Function** measures how well the neural network's predictions match the target labels. It provides a numerical value that the optimizer tries to minimize.

## Types of Loss Functions

### 1. Regression Losses
- **Mean Squared Error (MSE):** Measures the average squared difference. Sensitive to outliers.
  $$ MSE = \frac{1}{n} \sum (y - \hat{y})^2 $$
- **Mean Absolute Error (MAE):** Measures average absolute difference. Robust to outliers.
  $$ MAE = \frac{1}{n} \sum |y - \hat{y}| $$

### 2. Classification Losses
- **Binary Cross Entropy (BCE):** Used for binary classification.
  $$ L = -[y \log(\hat{y}) + (1-y) \log(1-\hat{y})] $$
- **Cross Entropy (CE):** Used for multi-class classification. Combines LogSoftmax and NLLLoss.

## Implementation Snippet (PyTorch)
```python
import torch.nn as nn

# Regression
criterion_mse = nn.MSELoss()

# Binary Classification
criterion_bce = nn.BCELoss() # Expects probabilities

# Multi-class Classification
criterion_ce = nn.CrossEntropyLoss() # Expects raw logits
```

## How to Choose?
- **Regression:** Use MSE for general tasks, MAE if you have many outliers.
- **Binary Classification:** BCE with a Sigmoid output.
- **Multi-class Classification:** Cross Entropy with a Softmax output (Logits).
- **Targeting Probabilities:** KL Divergence.
