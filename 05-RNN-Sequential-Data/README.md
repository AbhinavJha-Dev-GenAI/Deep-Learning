# 05. RNNs & Sequential Data 📜⏳

Recurrent Neural Networks (RNNs) are designed for data where the "order" matters, such as text, speech, or stock prices.

## 1. Why RNNs? ❓

Standard neural networks assume all inputs are independent. RNNs have a **Hidden State** ($h$) that acts as a "Memory," allowing the network to process sequences of varying lengths.

---

## 2. The Gated Revolution (LSTMs & GRUs) 💾

Standard RNNs struggle to remember long-term dependencies (Vanishing Gradient). Modern sequential models use **Gating Mechanisms**.

### LSTM (Long Short-Term Memory)
Uses three gates to control the flow of information:
- **Forget Gate**: What do we throw away?
- **Input Gate**: What new information do we add?
- **Output Gate**: What do we pass to the next state?

### GRU (Gated Recurrent Unit)
A simpler, faster version of LSTM with only two gates (Reset and Update).

---

## 3. Bidirectional RNNs 🔄

Instead of just looking at the past ($t-1$), a Bidirectional RNN looks at the future ($t+1$) simultaneously. 
- **Use Case**: Necessary for NLP (the meaning of a word often depends on the words following it).

---

## 🛠️ Essential Snippet (LSTM in PyTorch)

```python
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        _, (hn, _) = self.lstm(x) 
        # hn[-1] is the last hidden state
        return self.fc(hn[-1])
```

---

## 🚨 Summary
While LSTMs were the kings of NLP for a decade, they are computationally expensive because they process data **one step at a time**. This limitation eventually led to the invention of **Transformers**.
