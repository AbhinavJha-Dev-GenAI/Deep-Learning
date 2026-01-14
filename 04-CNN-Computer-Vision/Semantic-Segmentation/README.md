# Semantic Segmentation

**Semantic Segmentation** is the process of partitioning an image into multiple segments, assigning a class label to **every single pixel**.

## How it differs
- **Object Detection:** Bounding boxes.
- **Instance Segmentation:** Different labels for different individuals of the same class (e.g., Cat 1, Cat 2).
- **Semantic Segmentation:** Same label for all pixels of the same class (e.g., all pixels of all cats are labeled "Cat").

## Popular Architectures

### 1. U-Net
- Symmetric Encoder-Decoder structure with **Skip Connections**.
- Very popular in medical imaging.

### 2. DeepLabV3+
- Uses **Atrous (Dilated) Convolutions** to capture multi-scale context.

### 3. FCN (Fully Convolutional Network)
- Replaces fully connected layers with convolutions to support variable input sizes.

## Key Mechanics
- **Upsampling / Transposed Convolution:** Increasing the spatial resolution back to the original image size.
- **Skip Connections:** Helping the decoder recover fine-grained spatial details lost during downsampling.

## Loss Function
- **Pixel-wise Cross Entropy.**
- **Dice Loss:** Useful for handling class imbalance (especially in medical data).
