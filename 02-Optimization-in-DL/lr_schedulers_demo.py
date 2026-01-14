import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

def plot_schedulers():
    model = nn.Linear(10, 1)
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    
    epochs = 100
    
    # 1. StepLR
    s1 = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    # 2. ExponentialLR
    optimizer_exp = optim.SGD(model.parameters(), lr=0.1)
    s2 = optim.lr_scheduler.ExponentialLR(optimizer_exp, gamma=0.95)
    # 3. CosineAnnealing
    optimizer_cos = optim.SGD(model.parameters(), lr=0.1)
    s3 = optim.lr_scheduler.CosineAnnealingLR(optimizer_cos, T_max=epochs)

    lrs1, lrs2, lrs3 = [], [], []

    for _ in range(epochs):
        lrs1.append(s1.get_last_lr()[0])
        lrs2.append(s2.get_last_lr()[0])
        lrs3.append(s3.get_last_lr()[0])
        
        # In a real loop, you'd do optimizer.step() then scheduler.step()
        s1.step()
        s2.step()
        s3.step()

    plt.figure(figsize=(10, 6))
    plt.plot(lrs1, label='StepLR (step=30, gamma=0.1)')
    plt.plot(lrs2, label='ExponentialLR (gamma=0.95)')
    plt.plot(lrs3, label='CosineAnnealingLR (T_max=100)')
    
    plt.title("Learning Rate Schedulers over 100 Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = "lr_schedulers_demo.png"
    plt.savefig(save_path)
    print(f"Scheduler plot saved to {save_path}")

if __name__ == "__main__":
    plot_schedulers()
    
    # Demo ReduceLROnPlateau logic
    print("\n--- ReduceLROnPlateau Note ---")
    print("ReduceLROnPlateau requires a metric (like val_loss).")
    print("Example: scheduler.step(validation_loss)")
