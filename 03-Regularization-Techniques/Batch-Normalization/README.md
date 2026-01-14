# Batch Normalization

**Batch Normalization (BatchNorm)** is a technique to normalize the inputs of each layer so that they have a mean of zero and a variance of one.

## Why use BatchNorm?
- **Speed:** Accelerates training by allowing higher learning rates.
- **Stability:** Reduces internal covariate shift.
- **Regularization:** Adds a slight noise to the training process, acting as a regularizer.

## Mechanics
It normalizes the activations of a mini-batch:
$$ \hat{x} = \frac{x - E[x]}{\sqrt{Var(x) + \epsilon}} $$
Then it scales and shifts the result using learnable parameters $\gamma$ and $\beta$:
$$ y = \gamma \hat{x} + \beta $$

## Implementation Snippet (PyTorch)
```python
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(10, 50),
    nn.BatchNorm1d(50), # Applied after Linear, before Activation
    nn.ReLU(),
    nn.Linear(50, 1)
)

# For CNNs
# nn.BatchNorm2d(num_channels)
```

## Training vs Inference
- **Training:** Uses batch mean and variance.
- **Inference:** Uses running mean and variance tracked during training.
