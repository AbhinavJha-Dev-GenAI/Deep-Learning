# 01. Neural Network Fundamentals 🧠🧱

Neural Networks are the backbone of modern AI, designed to mimic the way biological neurons process information.

## 1. The Artificial Neuron (Perceptron) 🧬

An artificial neuron is a mathematical function that:
1.  Takes multiple **Inputs** ($x$).
2.  Multiplies them by **Weights** ($w$).
3.  Adds a **Bias** ($b$).
4.  Passes the total through an **Activation Function** ($\sigma$).

$$y = \sigma(\sum w_i x_i + b)$$

---

## 2. Activation Functions ⚡

Activation functions introduce **Non-linearity**, allowing the network to learn complex patterns instead of just simple lines.

| Function | Formula | Best For... |
| :--- | :--- | :--- |
| **ReLU** | $max(0, x)$ | Hidden layers (Standard Choice). |
| **Sigmoid** | $1 / (1 + e^{-x})$ | Binary classification output (0 to 1). |
| **Tanh** | Hyperbolic Tangent | Centered data (-1 to 1). |
| **Softmax** | Normalized Exponential | Multi-class classification output. |

---

## 3. Training: Backpropagation & Chain Rule 🔄

Training a neural network is an iterative process:
- **Forward Pass**: Data flows from input to output to get a prediction.
- **Loss Calculation**: Measuring how far off the prediction is from the truth (e.g., MSE).
- **Backward Pass (Backprop)**: Calculating the gradient (slope) of the loss with respect to every weight using the **Chain Rule**.
- **Update**: Adjusting the weights in the direction that reduces the loss.

---

## 🛠️ Essential Snippet (Simple MLP in PyTorch)

```python
import torch
import torch.nn as nn

# 1. Define the Architecture
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128) # Input to Hidden
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)  # Hidden to Output
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# 2. Initialize
model = SimpleNet()
print(model)
```

---

## 📊 Summary
Neural Networks are essentially "Universal Function Approximators." By stacking simple neurons and using backpropagation, they can learn to represent almost any mapping between inputs and outputs.
