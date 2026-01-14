import torch

def backprop_demo():
    """
    Manual vs Autograd Backpropagation for a single neuron:
    y = sigmoid(w*x + b)
    Loss = (y - target)^2
    """
    print("--- Backpropagation: Manual vs Autograd ---")
    
    # Inputs and Parameters
    x = torch.tensor([1.5])
    w = torch.tensor([0.8], requires_grad=True)
    b = torch.tensor([-0.5], requires_grad=True)
    target = torch.tensor([1.0])

    # 1. Forward Pass
    z = w * x + b
    y = torch.sigmoid(z)
    loss = (y - target)**2

    print(f"Forward: x={x.item()}, w={w.item()}, b={b.item()}")
    print(f"Result: y={y.item():.4f}, Target={target.item()}, Loss={loss.item():.4f}")

    # 2. Autograd Backward Pass
    loss.backward()
    autograd_dw = w.grad.item()
    autograd_db = b.grad.item()

    # 3. Manual Backward Pass (Chain Rule)
    # Loss = (y - target)^2
    # dLoss/dy = 2 * (y - target)
    # y = sigmoid(z)
    # dy/dz = y * (1 - y)
    # z = w*x + b
    # dz/dw = x
    # dz/db = 1
    
    dl_dy = 2 * (y - target)
    dy_dz = y * (1 - y)
    dz_dw = x
    dz_db = 1.0

    manual_dw = (dl_dy * dy_dz * dz_dw).item()
    manual_db = (dl_dy * dy_dz * dz_db).item()

    print("\n--- Gradient Comparison ---")
    print(f"Autograd dLoss/dw: {autograd_dw:.6f}")
    print(f"Manual   dLoss/dw: {manual_dw:.6f}")
    print(f"Difference: {abs(autograd_dw - manual_dw):.2e}")
    
    print(f"\nAutograd dLoss/db: {autograd_db:.6f}")
    print(f"Manual   dLoss/db: {manual_db:.6f}")
    print(f"Difference: {abs(autograd_db - manual_db):.2e}")

if __name__ == "__main__":
    backprop_demo()
