# Weight Initialization

**Weight Initialization** is the process of setting the initial values for the weights in a neural network. Proper initialization is crucial for ensuring that the network trains efficiently and avoids gradient-related problems.

## Why it matters?
If weights are initialized too small, signals passing through the network will eventually vanish. If they are too large, signals will explode.

## Common Initialization Strategies

### 1. Zero Initialization
- **Problem:** All neurons in a layer will learn the same features (Symmetry Problem).
- **Rule:** Never initialize weights to zero (except for biases).

### 2. Random Normal/Uniform
- Initializing with small random numbers.
- **Problem:** Can still lead to vanishing/exploding gradients in deep networks.

### 3. Xavier (Glorot) Initialization
- Designed for symmetric activation functions like **Tanh** or **Sigmoid**.
- Keeps the variance of inputs and outputs the same.
- **Formula:** $Var(W) = \frac{2}{n_{in} + n_{out}}$

### 4. He (Kaiming) Initialization
- Designed for **ReLU** activation functions.
- Accounts for the fact that half of the neurons are inactive with ReLU.
- **Formula:** $Var(W) = \frac{2}{n_{in}}$

## Implementation Snippet (PyTorch)
```python
import torch.nn as nn

linear = nn.Linear(10, 5)

# Xavier Initialization
nn.init.xavier_uniform_(linear.weight)

# He Initialization
nn.init.kaiming_normal_(linear.weight, mode='fan_in', nonlinearity='relu')

# Bias Initialization
nn.init.constant_(linear.bias, 0)
```

## Summary Recommendation
- Using **ReLU**? -> Use **He Initialization**.
- Using **Tanh/Sigmoid**? -> Use **Xavier Initialization**.
