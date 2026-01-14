# 04. CNNs & Computer Vision 👁️🖼️

Convolutional Neural Networks (CNNs) are specialized architectures for processing grid-like data, such as images.

## 1. Core Building Blocks 🧱

### Convolutions
Instead of looking at every pixel individually, a small **Kernel** (e.g., $3 \times 3$) slides over the image to detect local features like edges, corners, and textures.

### Pooling (Downsampling)
Reducing the spatial dimension of the feature maps to reduce computation and prevent overfitting.
- **Max Pooling**: Keeps the maximum value in a window (Detects the "strongest" feature).
- **Average Pooling**: Averages values (Smoother).

---

## 2. Evolution of Architectures 📈

*   **LeNet-5 (1998)**: The "Hello World" of CNNs. Used for digit recognition.
*   **AlexNet (2012)**: Proved that Deep CNNs + GPUs could win ImageNet by a landslide.
*   **VGG (2014)**: Introduced $3 \times 3$ convolutions as a modular standard.
*   **ResNet (2015)**: Introduced **Skip Connections** (Residuals), allowing for "Deep" networks (152+ layers) without the vanishing gradient problem.

---

## 3. Computer Vision Tasks 🎯

1.  **Classification**: "What is in this image?"
2.  **Detection**: "Where are the objects and what are they?" (YOLO, Faster R-CNN).
3.  **Segmentation**: "Which pixels belong to which object?" (U-Net, DeepLab).

---

## 🛠️ Essential Snippet (Residual Block in PyTorch)

```python
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1, stride)
            
    def forward(self, x):
        return nn.functional.relu(self.conv(x) + self.shortcut(x))
```

---

## 🌎 Summary
CNNs revolutionized the field by automating **Feature Engineering**. Instead of humans designing filters for "edges," the network learns which filters are most useful for the task.
