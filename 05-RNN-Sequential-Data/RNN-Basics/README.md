# RNN Basics

**Recurrent Neural Networks (RNNs)** are a class of neural networks designed to process **sequential data**. Unlike feedforward networks, RNNs have loops, allowing information to persist.

## Mechanics
At each time step $t$, the RNN takes an input $x_t$ and its own previous hidden state $h_{t-1}$ to produce a new hidden state $h_t$:
$$ h_t = \tanh(W_{xh} x_t + W_{hh} h_{t-1} + b) $$

## Why use RNNs?
- **Variable Length Inputs:** Can process sequences of different lengths (e.g., sentences).
- **Temporal Memory:** The hidden state acts as a memory that captures information about previous steps in the sequence.

## Use Cases
- Time-series prediction.
- Text generation.
- Speech recognition.

## The Fatal Flaw: Vanishing Gradients
In standard RNNs, when training on long sequences, the gradients transmitted back through time can vanish exponentially. This makes it impossible for the RNN to learn **long-term dependencies**.

## Implementation Snippet (PyTorch)
```python
import torch.nn as nn

# input_size, hidden_size, num_layers
rnn = nn.RNN(input_size=10, hidden_size=20, num_layers=1, batch_first=True)

# Input shape: (Batch, Seq, Features)
x = torch.randn(5, 3, 10)
h0 = torch.zeros(1, 5, 20) # Initial hidden state

out, h_n = rnn(x, h0)
```
