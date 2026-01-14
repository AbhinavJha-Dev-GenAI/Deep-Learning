# 02. Optimization in Deep Learning ⚡⚙️

Optimization is the process of finding the weight values that minimize the Loss Function. 

## 1. Gradient Descent Paradigms 🏔️

*   **Batch GD**: Uses the entire dataset to calculate one gradient (Slow but stable).
*   **Stochastic GD (SGD)**: Uses one random sample at a time (Fast but noisy).
*   **Mini-Batch GD**: The industry standard. Uses a small group (e.g., 32 or 64 samples) to balance speed and stability.

---

## 2. Modern Optimizers 🛠️

Generic SGD is often too slow or gets stuck in "local minima." We use algorithms that add **Momentum** or **Adaptivity**.

| Optimizer | Logic | Use Case |
| :--- | :--- | :--- |
| **Momentum** | Adds energy from past steps to "roll over" hills. | Faster convergence than SGD. |
| **RMSProp** | Scales the learning rate for each parameter. | Good for sequential data (RNNs). |
| **Adam** | Combines Momentum + RMSProp. | **The Default Choice** for 90% of tasks. |
| **AdamW** | Adam with better weight decay logic. | Standard for Training Transformers. |

---

## 3. Learning Rate Schedulers 📉

The "Learning Rate" is the most important hyperparameter.
- **High LR**: Fast learning, but might overshoot the goal.
- **Low LR**: Accurate, but takes forever to train.
**Schedulers** change the LR during training (e.g., starting high and getting lower as we approach the minimum).

---

## 🛠️ Essential Snippet (Optimizer Setup in PyTorch)

```python
import torch.optim as optim

# 1. Define Optimizer (Adam)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 2. Define Scheduler (Reduce LR if loss stops dropping)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

# 3. Inside the Training Loop
# ... after calculation ...
optimizer.zero_grad() # Clear old gradients
loss.backward()      # Compute new gradients
optimizer.step()     # Update weights
```

---

## 🧩 Pro-Tip: The "Vanishing Gradient"
If your gradients become too small (close to 0) during backprop, the weights stop updating. This is why we use **ReLU** instead of Sigmoid/Tanh in deep hidden layers!
