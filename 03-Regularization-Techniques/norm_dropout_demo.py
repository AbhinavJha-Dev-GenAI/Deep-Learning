import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

def get_model(use_bn=False, use_dropout=False):
    layers = []
    input_dim = 100
    for _ in range(5):
        layers.append(nn.Linear(input_dim, input_dim))
        if use_bn:
            layers.append(nn.BatchNorm1d(input_dim))
        layers.append(nn.ReLU())
        if use_dropout:
            layers.append(nn.Dropout(p=0.2))
    layers.append(nn.Linear(input_dim, 10))
    return nn.Sequential(*layers)

def train_and_track(model, X, y, epochs=100):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    losses = []
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        
    return losses

def compare_regularization():
    # Synthetic data
    X = torch.randn(128, 100)
    y = torch.randint(0, 10, (128,))

    # 1. Plain Model (High learning rate, deep)
    m1 = get_model(False, False)
    # 2. Model with BatchNorm
    m2 = get_model(True, False)
    # 3. Model with Dropout
    m3 = get_model(False, True)

    l1 = train_and_track(m1, X, y)
    l2 = train_and_track(m2, X, y)
    l3 = train_and_track(m3, X, y)

    plt.figure(figsize=(10, 6))
    plt.plot(l1, label='Baseline (Plain)')
    plt.plot(l2, label='With BatchNorm (Faster/Stable)')
    plt.plot(l3, label='With Dropout (Regularized)')
    
    plt.title("Effect of Regularization on Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = "norm_dropout_comparison.png"
    plt.savefig(save_path)
    print(f"Comparison plot saved to {save_path}")

if __name__ == "__main__":
    compare_regularization()
    
    # LayerNorm Demo
    print("\n--- LayerNorm Code ---")
    ln = nn.LayerNorm(100)
    x_sample = torch.randn(1, 100)
    print(f"Input Std: {x_sample.std().item():.4f}")
    print(f"LayerNorm Output Std: {ln(x_sample).std().item():.4f}")
