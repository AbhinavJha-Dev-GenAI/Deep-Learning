# LSTMs (Long Short-Term Memory)

**LSTM** is a special kind of RNN architecture explicitly designed to avoid the vanishing gradient problem and learn long-term dependencies.

## Key Innovation: The Cell State
The LSTM has a "cell state" (horizontal line running through the top) that acts like a conveyor belt, carrying information across long sequences with only minor linear interactions.

## The Three Gates
1. **Forget Gate:** Decides what information to discard from the cell state.
2. **Input Gate:** Decides which new information to store in the cell state.
3. **Output Gate:** Decides what information from the cell state to output as the hidden state.

## Why it works
Because the cell state is updated via addition rather than multiplication, gradients can flow back through many time steps without vanishing as easily as in standard RNNs.

## Implementation Snippet (PyTorch)
```python
import torch.nn as nn

# Similar API to RNN
lstm = nn.LSTM(input_size=10, hidden_size=20, num_layers=2, batch_first=True)

x = torch.randn(5, 3, 10)
# (num_layers, batch, hidden_size)
h0 = torch.zeros(2, 5, 20) # Hidden state
c0 = torch.zeros(2, 5, 20) # Cell state

out, (hn, cn) = lstm(x, (h0, c0))
```

## Comparison
- **Pros:** Excellent at capturing long-term patterns.
- **Cons:** More parameters than RNNs, computationally slower.
