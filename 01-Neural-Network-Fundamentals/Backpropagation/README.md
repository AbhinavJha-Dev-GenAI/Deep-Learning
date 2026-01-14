# Backpropagation

**Backpropagation** (Backward Propagation of Errors) is the central algorithm for training neural networks. It uses the **Chain Rule** from calculus to calculate the gradient of the loss function with respect to each weight.

## Overview
Training involves two passes:
1. **Forward Pass:** Input data goes through the network to produce a prediction.
2. **Backward Pass:** The error is calculated, and gradients are propagated backward from the output layer to the input layer.

## The Chain Rule
To update a weight $w$ in a hidden layer, we need $\frac{\partial Loss}{\partial w}$.
By Chain Rule:
$$ \frac{\partial Loss}{\partial w} = \frac{\partial Loss}{\partial Output} \cdot \frac{\partial Output}{\partial Activation} \cdot \frac{\partial Activation}{\partial w} $$

## Steps in Backpropagation
1. **Calculate the Error:** Use the loss function (e.g., MSE).
2. **Compute Gradients:** Calculate how much each weight contributed to the error.
3. **Update Weights:** Use an optimizer (e.g., SGD) to adjust weights: $w = w - \eta \cdot \nabla w$.

## Implementation Concept (PyTorch Autograd)
PyTorch automates backpropagation using `Autograd`.
```python
# Create tensors with gradient tracking
x = torch.tensor([1.0], requires_grad=True)
y = x ** 2

# Forward pass
loss = y - 5

# Backward pass
loss.backward()

# View gradient (dy/dx = 2x)
print(x.grad) # Output: 2.0
```

## Key Considerations
- **Computational Graph:** Neural networks are represented as graphs where nodes are operations.
- **Gradient Vanishing/Exploding:** During deep backprop, gradients can become too small or too large.
- **Automatic Differentiation:** Modern frameworks (PyTorch/TF) handle the complex math for you.
