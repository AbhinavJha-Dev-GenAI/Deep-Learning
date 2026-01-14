# Dropout

**Dropout** is a powerful regularization technique where neurons are randomly "dropped out" (set to zero) during training.

## Mechanics
- During each training step, each neuron has a probability $p$ of being temporarily ignored.
- This prevents neurons from over-relying on specific neighbors (co-adaptation).
- Effectively trains an ensemble of smaller sub-networks.

## Key Rules
1. **Training Mode:** Dropout is active.
2. **Evaluation Mode:** Dropout is disabled, and weights are scaled by $(1-p)$ to maintain the expected output magnitude.

## Implementation Snippet (PyTorch)
```python
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 100)
        self.dropout = nn.Dropout(p=0.5) # 50% chance of dropping
        self.out = nn.Linear(100, 1)
        
    def forward(self, x):
        x = torch.relu(self.fc(x))
        x = self.dropout(x)
        return self.out(x)

# CRITICAL
model.train() # Enable Dropout
model.eval()  # Disable Dropout
```

## Where to apply?
- Usually after activation functions in fully connected layers.
- For CNNs, **Spatial Dropout** or **DropBlock** is often preferred over standard Dropout.
