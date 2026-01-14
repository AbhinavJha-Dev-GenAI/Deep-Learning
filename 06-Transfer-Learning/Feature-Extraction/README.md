# Feature Extraction (Transfer Learning)

In **Feature Extraction**, we take a pre-trained model, "freeze" its weights, and use its output as features for a new classifier.

## How it works
1. **Load Pre-trained Model:** Load a model like ResNet-50.
2. **Freeze Layers:** Set `requires_grad = False` for all layers except the last one.
3. **Replace Classifier:** Remove the original output layer and add a new one that matches your target number of classes.
4. **Train Only the Head:** Only the weights of the new output layer are updated.

## Implementation Snippet (PyTorch)
```python
from torchvision import models
import torch.nn as nn

# 1. Load pre-trained model
model = models.resnet18(pretrained=True)

# 2. Freeze all parameters
for param in model.parameters():
    param.requires_grad = False

# 3. Replace the final fully connected layer
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10) # 10 new classes

# 4. Optimizer only updates the fc layer
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)
```

## When to use?
- When your dataset is very small.
- When the target task is very similar to the original task (e.g., classifying dogs vs. cats after training on ImageNet).
