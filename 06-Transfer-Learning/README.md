# 06. Transfer Learning 🏗️🧩

Transfer Learning is the art of taking a model trained on a massive dataset (like ImageNet or Wikipedia) and repurposing it for your specific, smaller dataset.

## 1. Why Transfer Learning? 💰

1.  **Less Data**: You don't need 1,000,000 images to train a classifier.
2.  **Less Compute**: No need for weeks of training on 8 GPUs.
3.  **Higher Accuracy**: Pre-trained models already have a deep understanding of low-level features (edges, shapes, grammar).

---

## 2. The Two Strategies ⚖️

### A. Feature Extraction
Freeze the entire pre-trained model and only train a new "Head" (the final layers).
- **Use Case**: When your dataset is very different from the pre-training data.

### B. Fine-Tuning
Unfreeze some (or all) layers of the pre-trained model and train the entire network with a very low learning rate.
- **Use Case**: When your dataset is similar to the pre-training data.

---

## 3. Popular Pre-trained Models 🌟

*   **Vision**: ResNet, EfficientNet, ViT (Vision Transformer).
*   **NLP**: BERT, RoBERTa, Llama.
*   **Audio**: Whisper.

---

## 🛠️ Essential Snippet (Fine-Tuning ResNet in PyTorch)

```python
from torchvision import models
import torch.nn as nn

# 1. Load Pre-trained Model
model = models.resnet50(weights="DEFAULT")

# 2. Freeze Parameters
for param in model.parameters():
    param.requires_grad = False

# 3. Replace the Head
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2) # e.g., For binary classification

# 4. Only train the Head
# optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
```

---

## 🌐 Summary
Don't reinvent the wheel! In production, $99\%$ of the time you should start with a pre-trained model. Transfer Learning is the most practical skill for a modern AI Engineer.
