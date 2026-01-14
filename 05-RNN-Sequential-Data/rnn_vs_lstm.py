import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

class SimpleModel(nn.Module):
    def __init__(self, mode, input_size, hidden_size):
        super().__init__()
        if mode == 'RNN':
            self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        else:
            self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :]) # Predict from last hidden state

def train_on_long_task(mode):
    # Task: Remember the first bit of a 50-length sequence
    seq_len = 50
    input_size = 1
    hidden_size = 16
    
    model = SimpleModel(mode, input_size, hidden_size)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    losses = []
    for _ in range(500):
        # Generate data: [batch, seq, 1]
        x = torch.randn(32, seq_len, input_size)
        # Target is just the first element of the sequence
        y = x[:, 0, :]
        
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        
    return losses

def compare_rnn_lstm():
    print("Training RNN (Standard)...")
    rnn_losses = train_on_long_task('RNN')
    print("Training LSTM...")
    lstm_losses = train_on_long_task('LSTM')
    
    plt.figure(figsize=(10, 6))
    plt.plot(rnn_losses, label='Standard RNN (Vanishing Gradients)')
    plt.plot(lstm_losses, label='LSTM (Gated Memory)')
    
    plt.title(f"RNN vs LSTM: Memory Task (Sequence Length 50)")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = "rnn_vs_lstm_memory.png"
    plt.savefig(save_path)
    print(f"Comparison plot saved to {save_path}")

if __name__ == "__main__":
    compare_rnn_lstm()
