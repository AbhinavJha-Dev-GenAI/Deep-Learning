# Layer Normalization

**Layer Normalization (LayerNorm)** is a normalization technique often used in **RNNs** and **Transformers**.

## Difference from Batch Norm
- **Batch Norm:** Normalizes across the batch (dependent on batch size).
- **Layer Norm:** Normalizes across the features of a single example (independent of batch size).

## Why use Layer Norm?
- **Small Batch Sizes:** Works perfectly even with batch size 1.
- **NLP Tasks:** Sequence lengths vary, and normalizing across the batch can be unstable. Layer Norm is the standard for LLMs and BERT.

## Implementation Snippet (PyTorch)
```python
import torch.nn as nn

# Normalize over the last dimension (features)
layer_norm = nn.LayerNorm(normalized_shape=512)

x = torch.randn(20, 10, 512) # [Batch, Seq, Features]
output = layer_norm(x)
```

## Summary Comparison
| Norm Type | Dimension Normalized | Best For |
| :--- | :--- | :--- |
| **Batch Norm** | Across the batch | CNNs, Computer Vision |
| **Layer Norm** | Across the features | RNNs, Transformers, NLP |
