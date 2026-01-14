# Gradient Descent

**Gradient Descent** is an optimization algorithm used to minimize the loss function by iteratively moving in the direction of the steepest descent—the negative of the gradient.

## Mechanics
The weights are updated by subtracting the product of the learning rate ($\eta$) and the gradient of the loss function ($\nabla L$):
$$ w = w - \eta \cdot \nabla L(w) $$

### Batch Gradient Descent
- Computes the gradient using the **entire dataset**.
- **Pros:** Stable updates, guaranteed to converge to the global minimum for convex functions.
- **Cons:** Very slow for large datasets, requires massive memory.

## Key Hyperparameter: Learning Rate ($\eta$)
- **Too Small:** Training takes forever.
- **Too Large:** Overshoots the minimum and may diverge.

## Implementation Concept
```python
# Simple manual update loop
for epoch in range(epochs):
    gradient = compute_gradient(data, params)
    params = params - learning_rate * gradient
```

## Challenges
- Getting stuck in local minima.
- Getting stuck in saddle points.
- Choosing the "perfect" learning rate.
