import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def compare_losses():
    # Simulated Ground Truth and Predictions
    y_true = torch.tensor([10.0])
    y_pred = torch.linspace(5, 15, 100) # Predictions ranging from 5 to 15

    mse_loss = nn.MSELoss(reduction='none')
    mae_loss = nn.L1Loss(reduction='none')
    huber_loss = nn.HuberLoss(reduction='none', delta=1.0)

    # Compute losses for each prediction point
    mse_vals = [mse_loss(p, y_true).item() for p in y_pred]
    mae_vals = [mae_loss(p, y_true).item() for p in y_pred]
    huber_vals = [huber_loss(p, y_true).item() for p in y_pred]

    plt.figure(figsize=(10, 6))
    plt.plot(y_pred.numpy(), mse_vals, label='MSE (L2) - Quadratic', lw=2)
    plt.plot(y_pred.numpy(), mae_vals, label='MAE (L1) - Linear', lw=2)
    plt.plot(y_pred.numpy(), huber_vals, label='Huber - Robust', linestyle='--', lw=2)

    plt.axvline(x=10, color='red', linestyle=':', label='Ground Truth (10)')
    plt.title("Comparison of Regression Loss Functions")
    plt.xlabel("Predicted Value")
    plt.ylabel("Loss Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = "loss_comparison_plot.png"
    plt.savefig(save_path)
    print(f"Regression plot saved to {save_path}")

def cross_entropy_demo():
    print("\n--- Cross Entropy Loss Demo ---")
    # 3-class classification
    criterion = nn.CrossEntropyLoss()
    
    # Raw scores (logits) from model
    logits = torch.tensor([[2.0, 1.0, 0.1]], requires_grad=True)
    target = torch.tensor([0]) # Target class is 0 (first class)
    
    loss = criterion(logits, target)
    print(f"Logits: {logits.detach().tolist()}")
    print(f"Target Class: {target.item()}")
    print(f"Cross Entropy Loss: {loss.item():.4f}")
    
    # Derivative of loss wrt logits
    loss.backward()
    print(f"Gradients wrt Logits: {logits.grad.tolist()}")

if __name__ == "__main__":
    compare_losses()
    cross_entropy_demo()
