# RMSProp

**RMSProp (Root Mean Square Propagation)** is an optimizer developed by Geoffrey Hinton. It was one of the first popular adaptive learning rate methods.

## Mechanics
Instead of using all previous gradients (like AdaGrad, which causes the learning rate to vanish), RMSProp uses an **exponentially decaying average** of squared gradients.

$$ E[g^2]_t = 0.9 E[g^2]_{t-1} + 0.1 g^2_t $$
$$ w = w - \frac{\eta}{\sqrt{E[g^2]_t + \epsilon}} g_t $$

## Key Features
- **Solves AdaGrad's learning rate decay problem:** It keeps the learning rate from getting too small too soon.
- **Great for RNNs:** Often works better than Adam in recurrent architectures.
- **Normalizes gradients:** It essentially scales the gradient based on its recent magnitude.

## Implementation Snippet (PyTorch)
```python
import torch.optim as optim

optimizer = optim.RMSprop(model.parameters(), lr=0.01, alpha=0.99)
```

## When to use?
- When training Recurrent Neural Networks (RNNs).
- When looking for an alternative to Adam that might converge differently.
