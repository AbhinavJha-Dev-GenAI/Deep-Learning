# 09. Deep Learning Interview Preparation 🧠🧱

Technical and scenario-based questions for Deep Learning Engineer roles.

## 1. Concepts & Architectures 📖

*   **Q: Why use ReLU over Sigmoid in hidden layers?**
    - *A:* ReLU avoids the **Vanishing Gradient** problem (its derivative is 1 for positive values) and is much faster to compute. Sigmoid "squashes" gradients to zero for large inputs.
*   **Q: Explain the Vanishing Gradient problem.**
    - *A:* During backprop, as we multiply derivatives through many layers, the gradient becomes smaller and smaller. Eventually, it reaches zero, and the weights in earlier layers stop updating.
*   **Q: What is a $1 \times 1$ Convolution?**
    - *A:* It is used to change the number of "channels" (depth) in an image without changing its spatial height and width. Crucial for Bottleneck layers in ResNet.

---

## 2. Training & Debugging 🛠️

*   **Q: Your model has 99% training accuracy but 60% validation accuracy. What's happening?**
    - *A:* The model is **Overfitting**. Solution: Add Dropout, Batch Norm, Weight Decay (L2), or more data.
*   **Q: What does Batch Normalization actually do?**
    - *A:* It normalizes the inputs to each layer to have 0 mean and 1 variance. This keeps the distribution of activations stable (Internal Covariate Shift), allowing for higher learning rates.
*   **Q: Adam vs SGD: Which to pick?**
    - *A:* Adam is the standard default because it has an adaptive learning rate per parameter. SGD with Momentum is often preferred in research to find "sharper" minima that generalize better, but it's much harder to tune.

---

## 3. Mathematical Intuition 🧪

*   **Q: Write the derivative of the Sigmoid function in terms of itself.**
    - *A:* $\sigma'(x) = \sigma(x) \cdot (1 - \sigma(x))$
*   **Q: Explain the "Dead ReLU" problem.**
    - *A:* If a neuron's input is always negative, its gradient will always be 0. It will never update again and is effectively "dead." Solution: Use **Leaky ReLU**.

---

## 🎯 DL Cheat Sheet
1. **Best Activation**: ReLU (Hidden), Softmax (Output).
2. **Best Optimizer**: Adam / AdamW.
3. **Best Regularization**: Dropout + BatchNorm.
4. **Best Architecture**: ResNet (Vision), Transformer (NLP).
