# CNN Architectures

Over the years, several landmark architectures have shaped the field of Computer Vision.

## Landmark Models

### 1. LeNet-5 (1998)
- Designed by Yann LeCun for handwritten digit recognition.
- Introduced the concept of Convolution -> Pooling -> FC.

### 2. AlexNet (2012)
- Re-ignited interest in Deep Learning by winning ImageNet.
- Used ReLU, Dropout, and GPU acceleration.

### 3. VGG (2014)
- Focused on depth using small $3 \times 3$ filters repeatedly.
- Easy to understand but computationally heavy.

### 4. ResNet (2015) - Residual Networks
- Introduced **Skip Connections** (Residuals) to solve the Vanishing Gradient problem.
- Allowed for training networks with hundreds of layers (ResNet-50, ResNet-101).

### 5. EfficientNet (2019)
- Uses **Compound Scaling** to balance depth, width, and resolution.
- Achieves state-of-the-art accuracy with much fewer parameters.

## Which to choose?
- **Fast Training:** ResNet-18 or MobileNet.
- **State-of-the-Art Accuracy:** EfficientNet-B7 or Vision Transformers (ViT).
- **Embedded/Mobile:** MobileNetV2, ShuffleNet.

## Implementation (Using Pre-trained Weights)
```python
from torchvision import models

# Load a pre-trained ResNet-50
model = models.resnet50(pretrained=True)

# Replace the last layer for your specific task
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)
```
