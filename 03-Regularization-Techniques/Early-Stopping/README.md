# Early Stopping

**Early Stopping** is a regularization method used to stop the training process before the model starts overfitting.

## How it works
- Monitors the model's performance on a **Validation Set**.
- If the validation loss stops improving for a certain number of epochs (called `patience`), training is terminated.
- The weights from the best epoch are kept.

## Why use it?
- **Prevents Overfitting:** Stops training as soon as the model begins to learn noise from the training set.
- **Saves Resources:** Prevents wasted compute on redundant training epochs.

## Implementation Snippet (PyTorch)
PyTorch doesn't have a built-in EarlyStopping class, but it's easy to implement manually or using **Lightning**.
```python
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

# Usage in loop
early_stopper = EarlyStopping(patience=5)
for epoch in range(100):
    val_loss = validate()
    early_stopper(val_loss)
    if early_stopper.early_stop:
        print("Stopping early!")
        break
```

## Best Practice
- Always use a `patience` value that accounts for short-term fluctuations in validation loss.
- Often used alongside **Learning Rate Schedulers**.
