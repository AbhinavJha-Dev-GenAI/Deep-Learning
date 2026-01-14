# Learning Rate Schedulers

A **Learning Rate Scheduler** adjusts the learning rate during training based on the epoch number or validation performance.

## Why use Schedulers?
Initially, a high learning rate helps the model explore the loss landscape quickly. As training progresses, a smaller learning rate is needed to "settle" into the global minimum.

## Common Schedulers

### 1. StepLR
Decays the learning rate by a factor $\gamma$ every $K$ epochs.
```python
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
```

### 2. MultiStepLR
Decays the learning rate at specific milestones.
```python
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 80], gamma=0.1)
```

### 3. ExponentialLR
Decays the learning rate exponentially.
```python
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
```

### 4. ReduceLROnPlateau
Reduces the learning rate when a metric (e.g., validation loss) has stopped improving.
```python
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
```

### 5. CosineAnnealingLR
Follows a cosine curve to decay the learning rate.
```python
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
```

## Best Practice
1. Start with a fixed learning rate ($10^{-3}$ for Adam).
2. If training plateaus, introduce **ReduceLROnPlateau** or a **Cosine Annealing** strategy.
