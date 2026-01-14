# Activation Functions

**Activation Functions** decide whether a neuron should be activated or not by calculating the weighted sum and further adding bias with it. They introduce **non-linearity** into the output of a neuron.

## Why Non-Linearity?
Without non-linear activation functions, no matter how many layers a neural network has, it would behave like a single-layer perceptron because the composition of linear functions is still a linear function.

## Common Activation Functions

### 1. ReLU (Rectified Linear Unit)
- **Formula:** $f(x) = \max(0, x)$
- **Pros:** Fast to compute, avoids vanishing gradient for positive values.
- **Cons:** "Dying ReLU" problem where neurons get stuck at 0.

### 2. Sigmoid
- **Formula:** $\sigma(x) = \frac{1}{1 + e^{-x}}$
- **Range:** (0, 1)
- **Use Case:** Binary classification output layer.
- **Cons:** Vanishing gradient problem, output is not zero-centered.

### 3. Tanh (Hyperbolic Tangent)
- **Formula:** $\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$
- **Range:** (-1, 1)
- **Pros:** Zero-centered output.
- **Cons:** Vanishing gradient problem.

### 4. Modern Variants (GELU, Swish)
- **GELU:** Used in Transformers (BERT, GPT).
- **Swish:** $x \cdot \sigma(\beta x)$, developed by Google.

## Implementation Snippet (PyTorch)
```python
import torch.nn.functional as F

# Using functions
x = torch.randn(2)
relu_out = F.relu(x)
sigmoid_out = torch.sigmoid(x)

# Using as modules in nn.Sequential
model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 1),
    nn.Sigmoid()
)
```

## Comparison Table
| Function | Range | Zero-Centered | Main Advantage |
| :--- | :--- | :--- | :--- |
| Sigmoid | (0, 1) | No | Probability output |
| Tanh | (-1, 1) | Yes | Better than sigmoid for hidden layers |
| ReLU | [0, âˆž) | No | Efficiency, sparseness |
| Leaky ReLU| (-âˆž, âˆž)| No | Solves "Dying ReLU" |
