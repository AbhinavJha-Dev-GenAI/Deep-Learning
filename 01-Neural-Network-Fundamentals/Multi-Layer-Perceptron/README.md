# Multi-Layer Perceptron (MLP)

A **Multi-Layer Perceptron (MLP)** is a class of feedforward artificial neural network. It consists of at least three layers of nodes: an input layer, a hidden layer, and an output layer.

## Overview
MLPs were designed to overcome the limitations of single-layer perceptrons, specifically the inability to solve non-linearly separable problems like XOR. By adding "hidden layers" and non-linear activation functions, MLPs can approximate any continuous function (Universal Approximation Theorem).

## Key Mechanics
1. **Feedforward Propagation:** Data moves from input to output.
2. **Hidden Layers:** Extract complex features using non-linear transformations.
3. **Activation Functions:** (ReLU, Sigmoid, Tanh) introduce non-linearity.
4. **Softmax Output:** Often used for multi-class classification to produce probabilities.

## Learning Process
MLPs use **Backpropagation** to learn. The error is calculated at the output and propagated backwards through the network to update weights using Gradient Descent.

## Use Cases
- Tabular data classification.
- Simple regression tasks.
- Feature extraction in larger architectures.

## Implementation Snippet (PyTorch)
```python
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )
        
    def forward(self, x):
        return self.network(x)

# Example: 10 inputs -> 64 hidden nodes -> 3 output classes
model = MLP(10, 64, 3)
```

## Pros and Cons
| Pros | Cons |
| :--- | :--- |
| Can learn non-linear relationships | Prone to overfitting on small datasets |
| Universal Function Approximator | Computationally expensive as layers increase |
| Scales well with large datasets | "Black box" nature—hard to interpret |
