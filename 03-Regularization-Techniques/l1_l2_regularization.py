import torch
import torch.nn as nn
import torch.optim as optim

def l1_l2_demo():
    model = nn.Sequential(
        nn.Linear(10, 50),
        nn.ReLU(),
        nn.Linear(50, 1)
    )

    # 1. L2 Regularization (Weight Decay)
    # Built directly into the optimizer
    optimizer_l2 = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    print("PyTorch Weight Decay (L2) configured in Adam.")

    # 2. L1 Regularization (Manual)
    # Must be added manually to the loss function
    l1_lambda = 0.01
    
    x = torch.randn(5, 10)
    target = torch.randn(5, 1)
    
    # Forward
    output = model(x)
    mse_loss = nn.MSELoss()(output, target)
    
    # Calculate L1 penalty
    l1_penalty = sum(p.abs().sum() for p in model.parameters())
    
    total_loss = mse_loss + l1_lambda * l1_penalty
    
    print(f"MSE Loss: {mse_loss.item():.4f}")
    print(f"L1 Penalty: {l1_penalty.item():.4f}")
    print(f"Total Loss: {total_loss.item():.4f}")

    # 3. Observing Weight Magnitudes
    def get_sparsity(m):
        total = 0
        zeros = 0
        for p in m.parameters():
            total += p.numel()
            zeros += torch.sum(p.abs() < 1e-3).item()
        return zeros / total

    print(f"Initial Sparsity: {get_sparsity(model) * 100:.2f}%")

if __name__ == "__main__":
    l1_l2_demo()
