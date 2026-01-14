import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def demonstrate_gradient_issues():
    """
    Demonstrates Vanishing Gradients (Sigmoid) and Exploding Gradients (Large weights).
    """
    input_dim = 1
    depth = 50
    
    # 1. Vanishing Gradients Demo
    vanishing_model = []
    for _ in range(depth):
        layer = nn.Linear(input_dim, input_dim)
        # Using Sigmoid and standard init often causes vanishing
        vanishing_model.append(layer)
        vanishing_model.append(nn.Sigmoid())
    
    # 2. Exploding Gradients Demo
    exploding_model = []
    for _ in range(depth):
        layer = nn.Linear(input_dim, input_dim)
        # Manually set large weights
        nn.init.constant_(layer.weight, 2.0)
        exploding_model.append(layer)
        exploding_model.append(nn.ReLU())

    x = torch.tensor([[1.0]], requires_grad=True)
    
    # Track gradient norms through depth
    v_grads = []
    e_grads = []
    
    # Forward/Backward for Vanishing
    out = x
    for layer in vanishing_model:
        out = layer(out)
    out.backward()
    
    # We look at the gradient of the FIRST layer wrt the output
    # Since we can't easily hook, we'll simulate the chain product
    
    print("--- Gradient Issue Simulation ---")
    
    v_val = 0.25 # Max derivative of sigmoid
    e_val = 2.0  # Weight value
    
    v_curve = [v_val**i for i in range(depth)]
    e_curve = [min(e_val**i, 1e6) for i in range(depth)] # Cap for plotting

    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(v_curve, color='orange')
    plt.yscale('log')
    plt.title("Vanishing Gradient Effect\n(0.25^depth)")
    plt.xlabel("Layer Depth")
    plt.ylabel("Gradient Scale (Log)")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(e_curve, color='red')
    plt.yscale('log')
    plt.title("Exploding Gradient Effect\n(2.0^depth)")
    plt.xlabel("Layer Depth")
    plt.ylabel("Gradient Scale (Log)")
    plt.grid(True)

    plt.tight_layout()
    save_path = "gradient_issues_demo.png"
    plt.savefig(save_path)
    print(f"Gradient plot saved to {save_path}")

def gradient_clipping_demo():
    print("\n--- Gradient Clipping Demo ---")
    param = torch.tensor([100.0], requires_grad=True)
    # Simulate a huge gradient
    param.grad = torch.tensor([500.0])
    
    print(f"Original Gradient: {param.grad.item()}")
    nn.utils.clip_grad_norm_([param], max_norm=1.0)
    print(f"Clipped Gradient (Norm): {param.grad.item()}")

if __name__ == "__main__":
    demonstrate_gradient_issues()
    gradient_clipping_demo()
