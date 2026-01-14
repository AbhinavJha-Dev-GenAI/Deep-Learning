# Image Classification

**Image Classification** is the task of assigning a label from a predefined set of categories to an entire input image.

## Overview
It is one of the core problems in Computer Vision. A model takes an image as input and outputs a probability distribution over the classes.

## Workflow
1. **Preprocessing:** Resize, normalize, and Augment data.
2. **Backbone:** Use a CNN (e.g., ResNet, VGG) to extract features.
3. **Head:** A Global Average Pooling layer followed by a Fully Connected layer with Softmax.
4. **Loss Function:** Usually **Cross-Entropy Loss**.

## Data Augmentation
Techniques to artificially increase dataset size and prevent overfitting:
- Random Cropping / Resizing.
- Horizontal / Vertical Flips.
- Color Jitter (brightness, contrast, saturation).
- Rotation and Shear.

## Implementation Snippet (PyTorch)
```python
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

## Key Metrics
- **Accuracy:** General performance.
- **Top-5 Accuracy:** If the correct class is among the top 5 predictions.
- **Precision / Recall / F1-Score:** Important for imbalanced datasets.
