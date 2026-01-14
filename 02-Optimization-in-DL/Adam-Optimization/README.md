# Adam Optimization

**Adam (Adaptive Moment Estimation)** is currently the most popular optimization algorithm in Deep Learning. It combines the benefits of **AdaGrad** and **RMSProp**.

## How it Works
Adam maintains two moving averages for each parameter:
1. **$m_t$ (First Moment):** Moving average of gradients (like Momentum).
2. **$v_t$ (Second Moment):** Moving average of the *squared* gradients (like RMSProp).

$$ w_{t+1} = w_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t $$

## Why use Adam?
- **Adaptive Learning Rates:** It automatically adjusts the learning rate for each parameter.
- **Robust:** Works well across many architectures (CNNs, Transformers, MLPs).
- **Default Choice:** Usually the starting point for any new DL project.

## Implementation Snippet (PyTorch)
```python
import torch.optim as optim

# Learning rate 1e-3 is often a good default for Adam
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
```

## Key Hyperparameters
- **$\eta$ (Learning Rate):** Usually $10^{-3}$ or $3 \cdot 10^{-4}$.
- **$\beta_1$:** Exponential decay for the first moment (default 0.9).
- **$\beta_2$:** Exponential decay for the second moment (default 0.999).
- **$\epsilon$:** Small constant to prevent division by zero (default $10^{-8}$).
