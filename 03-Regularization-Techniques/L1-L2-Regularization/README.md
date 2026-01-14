# L1 & L2 Regularization

Regularization is a technique used to prevent **overfitting** by penalizing large weights in the model.

## L1 Regularization (Lasso)
Adds the **absolute value** of weights as a penalty term to the loss function.
$$ L_{reg} = L + \lambda \sum |w| $$
- **Effect:** Encourages **sparsity** (sets many weights exactly to zero).
- **Use Case:** Feature selection.

## L2 Regularization (Ridge / Weight Decay)
Adds the **squared value** of weights as a penalty term.
$$ L_{reg} = L + \lambda \sum w^2 $$
- **Effect:** Keeps weights small but rarely zero. Distributes error across all features.
- **Use Case:** General-purpose regularization.

## Implementation Snippet (PyTorch)
In PyTorch, L2 regularization is built into the optimizer as `weight_decay`.
```python
# L2 Regularization (Weight Decay)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

# L1 Regularization (Manual)
l1_lambda = 0.001
l1_norm = sum(p.abs().sum() for p in model.parameters())
loss = criterion(pred, target) + l1_lambda * l1_norm
```

## Comparison
| Feature | L1 (Lasso) | L2 (Ridge) |
| :--- | :--- | :--- |
| **Penalty** | $|w|$ | $w^2$ |
| **Weights** | Can become zero | Small, but non-zero |
| **Solution** | Sparse | Dense |
| **Robustness**| More robust to outliers | Less robust |
