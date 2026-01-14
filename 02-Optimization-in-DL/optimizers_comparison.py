import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

def rosenbrock(x, y):
    """The Rosenbrock function: a classic optimization test case."""
    return (1 - x)**2 + 100 * (y - x**2)**2

def compare_optimizers():
    optimizers_config = {
        "SGD": lambda p: optim.SGD(p, lr=0.001),
        "Momentum": lambda p: optim.SGD(p, lr=0.001, momentum=0.9),
        "RMSProp": lambda p: optim.RMSprop(p, lr=0.01),
        "Adam": lambda p: optim.Adam(p, lr=0.1)
    }

    results = {}

    for name, opt_func in optimizers_config.items():
        # Starting point
        params = torch.tensor([-2.0, 2.0], requires_grad=True)
        optimizer = opt_func([params])
        
        history = []
        for _ in range(100):
            optimizer.zero_grad()
            loss = rosenbrock(params[0], params[1])
            history.append(loss.item())
            loss.backward()
            optimizer.step()
        
        results[name] = history

    # Plotting
    plt.figure(figsize=(10, 6))
    for name, history in results.items():
        plt.plot(history, label=name)

    plt.yscale('log')
    plt.title("Optimizer Comparison on Rosenbrock Function")
    plt.xlabel("Iterations")
    plt.ylabel("Loss (Log Scale)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = "optimizer_comparison.png"
    plt.savefig(save_path)
    print(f"Optimizer comparison plot saved to {save_path}")

if __name__ == "__main__":
    compare_optimizers()
    
    # Print a summary of PyTorch optimizer usage
    print("\n--- PyTorch Optimizer Summary ---")
    print("1. optim.SGD(params, lr=0.01) -> Simple gradient descent")
    print("2. optim.Adam(params, lr=1e-3) -> Adaptive, good default")
    print("3. optim.RMSprop(params, lr=1e-2) -> Good for RNNs")
