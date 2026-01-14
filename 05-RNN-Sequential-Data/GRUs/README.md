# GRUs (Gated Recurrent Units)

The **GRU** is a simpler version of the LSTM, introduced by Cho et al. in 2014. It aims to provide similar performance to LSTMs but with fewer parameters.

## Key Differences from LSTM
1. **No Cell State:** It only has hidden states.
2. **Two Gates:** It combines the forget and input gates into a single **Update Gate**. It also has a **Reset Gate**.
3. **Simplicity:** Because it has fewer gates, it is faster to train and requires less data to converge.

## The Gates
- **Update Gate:** Determines how much of the past information needs to be passed along to the future.
- **Reset Gate:** Determines how much of the past information to forget.

## Implementation Snippet (PyTorch)
```python
import torch.nn as nn

gru = nn.GRU(input_size=10, hidden_size=20, num_layers=1, batch_first=True)

x = torch.randn(5, 3, 10)
h0 = torch.zeros(1, 5, 20)

out, hn = gru(x, h0)
```

## GRU vs LSTM
| Feature | LSTM | GRU |
| :--- | :--- | :--- |
| **Parameters**| More | Fewer (Faster) |
| **Performance**| Generally slightly better for very long sequences | Similar performance for most tasks |
| **State** | Hidden + Cell | Hidden only |
