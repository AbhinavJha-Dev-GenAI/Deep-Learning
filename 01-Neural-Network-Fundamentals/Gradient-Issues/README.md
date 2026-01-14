# Gradient Issues: Vanishing & Exploding

As neural networks get deeper, the gradients can become extremely small or extremely large during backpropagation, making training difficult or impossible.

## 1. Vanishing Gradient Problem
**What is it?** 
When gradients become so small (close to 0) that the weights in early layers barely update. The network stops learning.

**Causes:**
- Deep networks with Sigmoid or Tanh activations.
- Poor weight initialization.

**Solutions:**
- Use **ReLU** or its variants (Leaky ReLU).
- Use **Batch Normalization**.
- Use **Residual Connections** (ResNets).
- Proper Weight Initialization (He/Xavier).

## 2. Exploding Gradient Problem
**What is it?**
When gradients accumulate and become very large, leading to huge updates to weights. This makes the loss "explode" to `NaN` or move erratically.

**Causes:**
- Large weights.
- High learning rate.
- Deep Recurrent Neural Networks (RNNs).

**Solutions:**
- **Gradient Clipping:** Capping the maximum value of gradients.
- Batch Normalization.
- Reduced Learning Rate.

## Summary Table
| Issue | Symptom | Primary Solution |
| :--- | :--- | :--- |
| **Vanishing** | Training is very slow, loss plateaus early | ReLU, Skip Connections, Batch Norm |
| **Exploding** | Loss becomes `NaN`, weights change wildly | Gradient Clipping, Batch Norm |
