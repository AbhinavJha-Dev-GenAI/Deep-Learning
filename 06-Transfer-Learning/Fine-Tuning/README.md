# Fine-Tuning (Transfer Learning)

**Fine-Tuning** involves unfreezing some or all of the weights of a pre-trained model and training them on a new dataset with a very low learning rate.

## Strategies
1. **Partial Fine-Tuning:** Freeze the early layers (which learn general features) and fine-tune only the later layers (which learn task-specific features).
2. **Full Fine-Tuning:** Fine-tune the entire network. Requires more data and more compute.

## Why a Low Learning Rate?
We use a very small learning rate (e.g., $10^{-5}$) to avoid "destroying" the useful weights the model has already learned during pre-training.

## Implementation Snippet (PyTorch)
```python
# Unfreeze all parameters
for param in model.parameters():
    param.requires_grad = True

# Use a VERY small learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
```

## When to use?
- When you have a large dataset.
- When the target task is significantly different from the original task.
- To squeeze out every bit of performance.
