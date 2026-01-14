# Perceptron

The **Perceptron** is the fundamental building block of neural networks. It is a type of artificial neuron that takes multiple binary inputs and produces a single binary output based on a weighted sum.

## Overview
Developed by Frank Rosenblatt in 1958, the perceptron is the simplest form of a neural network used for binary classification. It models a single neuron that remains inactive unless its input signals exceed a certain threshold.

## Key Mechanics
The perceptron calculates the weighted sum of its inputs and applies a step function:

$$ y = \begin{cases} 1 & \text{if } \sum_{i=1}^{n} w_i x_i + b > 0 \\ 0 & \text{otherwise} \end{cases} $$

- **Inputs ($x_i$):** Features of the data.
- **Weights ($w_i$):** Importance assigned to each input.
- **Bias ($b$):** An offset that allows the activation function to shift.
- **Activation Function:** Usually a Step Function (Heaviside).

## Learning Rule
The perceptron learns by adjusting its weights based on the error:
$$ w_{new} = w_{old} + \eta (y_{target} - y_{predicted}) x $$
where $\eta$ is the learning rate.

## Use Cases
- Simple binary classification (e.g., Spam vs. Not Spam).
- Linearly separable data.
- Core component of Multi-Layer Perceptrons.

## Implementation Snippet (PyTorch)
```python
import torch
import torch.nn as nn

class Perceptron(nn.Module):
    def __init__(self, input_size):
        super(Perceptron, self).__init__()
        self.fc = nn.Linear(input_size, 1)
        
    def forward(self, x):
        # Heaviside step function approximation
        return torch.where(self.fc(x) > 0, 1.0, 0.0)

# Example usage
model = Perceptron(input_size=2)
x = torch.tensor([0.5, -0.2])
print(model(x))
```

## Pros and Cons
| Pros | Cons |
| :--- | :--- |
| Extremely simple and fast | Can only learn linearly separable patterns |
| Computationally efficient | Fails on the XOR problem |
| Foundation for Deep Learning | Hard step function is not differentiable |
