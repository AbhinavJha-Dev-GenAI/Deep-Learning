# Bidirectional RNNs

**Bidirectional RNNs** process the input sequence in both forward and backward directions, allowing the network to have context from both the past and the future at any given time step.

## Mechanics
- Consists of two RNNs: one **Forward RNN** and one **Backward RNN**.
- The outputs of both are typically concatenated or summed at each time step.
- Output $y_t$ depends on $h_t^{forward}$ and $h_t^{backward}$.

## Why use Bidirectional RNNs?
In many tasks like Natural Language Processing (NLP), the meaning of a word often depends on the words that come *after* it, not just the ones before it.
- **Example:** "He is going to the bank to deposit money" vs. "He is going to the bank of the river."

## Implementation Snippet (PyTorch)
```python
import torch.nn as nn

# Simply set bidirectional=True
lstm = nn.LSTM(input_size=10, hidden_size=20, bidirectional=True, batch_first=True)

x = torch.randn(5, 3, 10)
out, (hn, cn) = lstm(x)

# out shape: (batch, seq, hidden_size * 2)
# hidden_size * 2 because of forward and backward concatenation
```

## Constraints
- **Not for real-time:** Cannot be used for online/real-time prediction (where you don't have the future data yet).
- **Offline tasks:** Perfect for translation, sentiment analysis, and named entity recognition.
