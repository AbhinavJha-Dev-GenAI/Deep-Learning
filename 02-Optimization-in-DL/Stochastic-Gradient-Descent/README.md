# Stochastic Gradient Descent (SGD)

**Stochastic Gradient Descent (SGD)** is a variation of gradient descent that updates parameters for **each training example** or a small **mini-batch**, rather than the entire dataset.

## Types of SGD

### 1. Pure SGD
- Updates weights after **every single example**.
- Extremely noisy updates, but can escape local minima.

### 2. Mini-Batch SGD (Most Common)
- Updates weights after a small batch of examples (e.g., 32, 64, 128).
- Strikes a balance between the stability of Batch GD and the speed of Pure SGD.

## SGD with Momentum
Traditional SGD oscillates across the "slopes" of the loss landscape. **Momentum** helps accelerate SGD in the relevant direction and dampens oscillations by using a moving average of gradients.
$$ v_t = \gamma v_{t-1} + \eta \nabla L(w) $$
$$ w = w - v_t $$

## Implementation Snippet (PyTorch)
```python
import torch.optim as optim

# Standard SGD
optimizer = optim.SGD(model.parameters(), lr=0.01)

# SGD with Momentum
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
```

## Pros and Cons
| Pros | Cons |
| :--- | :--- |
| Faster than Batch GD | Frequent updates cause high variance |
| Can handle massive datasets | Noise can make convergence difficult |
| Memory efficient | Hard to tune the learning rate |
