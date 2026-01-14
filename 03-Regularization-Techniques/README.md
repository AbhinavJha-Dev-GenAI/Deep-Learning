# 03. Regularization Techniques 🛡️⚖️

Regularization is the set of techniques used to prevent **Overfitting**—where a model performs perfectly on training data but fails on new, unseen data.

## 1. Weight Decay (L1 & L2) 📉

Forces the network to keep weights small, which makes the model "simpler."
- **L2 Regularization (Ridge)**: Adds the *square* of weights to the loss. This is the most common.
- **L1 Regularization (Lasso)**: Adds the *absolute value* of weights. This can drive some weights to exactly zero (Feature Selection).

---

## 2. Dropout: The "Random" Defense 🎲

During training, randomly "turn off" a percentage (e.g., 20% or 50%) of neurons in each layer for every batch.
- **Why?**: Forces the network to learn redundant representations so it's not overly dependent on any single neuron.

---

## 3. Batch Normalization (BatchNorm) 🧪

Normalizes the outputs of each layer so they have a mean of 0 and a variance of 1.
- **Benefits**: Stabilizes training, allows for much higher learning rates, and acts as a slight regularizer.

---

## 4. Early Stopping 🛑

Stop training the moment the **Validation Loss** starts increasing, even if the training loss is still going down. This prevents the model from "memorizing" the noise in the training set.

---

## 🛠️ Essential Snippet (Dropout & BatchNorm in PyTorch)

```python
import torch.nn as nn

class RegularizedNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(784, 512),
            nn.BatchNorm1d(512), # Normalize before activation
            nn.ReLU(),
            nn.Dropout(p=0.5),    # Randomly shut down 50%
            nn.Linear(512, 10)
        )
        
    def forward(self, x):
        return self.layers(x)
```

---

## ⚖️ Summary
Regularization is about the **Bias-Variance Tradeoff**. A perfectly regularized model might have slightly higher training error but will be far more robust and reliable in the real world.
