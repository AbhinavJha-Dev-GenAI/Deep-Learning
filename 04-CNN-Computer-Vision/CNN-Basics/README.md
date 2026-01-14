# CNN Basics

**Convolutional Neural Networks (CNNs)** are a specialized type of neural network designed for processing structured grid data, such as images.

## Core Components

### 1. Convolution Layer ($Conv2d$)
- Uses **filters** (kernels) to slide across the input image.
- Performs element-wise multiplication and summation.
- Extracts spatial features like edges, textures, and patterns.

### 2. Pooling Layer ($MaxPool2d$, $AvgPool2d$)
- Reduces the spatial dimensions (width, height) of the feature maps.
- Makes the model invariant to small translations in the image.
- Reduces the number of parameters and computation.

### 3. Fully Connected (FC) Layer
- Takes the flattened feature map and produces the final classification or regression output.

## Key Hyperparameters
- **Kernel Size:** Size of the filter (e.g., $3 \times 3$, $5 \times 5$).
- **Stride:** How many pixels the filter moves at a time.
- **Padding:** Adding zeros to the border to maintain spatial dimensions.
- **Dilation:** Spacing between kernel elements (for receptive field expansion).

## Why CNNs instead of MLPs?
- **Parameter Sharing:** A filter is applied to the whole image.
- **Sparsity of Connections:** Each output value depends only on a small local region of the input.
- **Translation Invariance:** Can recognize a feature regardless of where it is in the image.

## Implementation Snippet (PyTorch)
```python
import torch.nn as nn

conv_layers = nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2)
)
```
