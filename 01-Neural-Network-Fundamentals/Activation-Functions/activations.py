import torch
import matplotlib.pyplot as plt
import numpy as np

def plot_activations():
    x = torch.linspace(-5, 5, 200, requires_grad=True)

    activations = {
        "ReLU": torch.relu,
        "Sigmoid": torch.sigmoid,
        "Tanh": torch.tanh,
        "Leaky ReLU": lambda x: torch.nn.functional.leaky_relu(x, 0.1)
    }

    plt.figure(figsize=(12, 8))

    for i, (name, func) in enumerate(activations.items(), 1):
        plt.subplot(2, 2, i)
        
        # Forward pass
        y = func(x)
        plt.plot(x.detach().numpy(), y.detach().numpy(), label=f'{name}(x)', color='blue')
        
        # Gradient calculation
        y.sum().backward(retain_graph=True)
        plt.plot(x.detach().numpy(), x.grad.numpy(), label=f"d{name}/dx", linestyle='--', color='red')
        
        # Reset gradient for next function
        x.grad.zero_()
        
        plt.title(f"{name} and its Gradient")
        plt.grid(True)
        plt.legend()
        plt.axhline(0, color='black', lw=1)
        plt.axvline(0, color='black', lw=1)

    plt.tight_layout()
    plt.suptitle("Activation Functions and their Derivatives", fontsize=16, y=1.02)
    
    save_path = "activation_functions_plot.png"
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    # plt.show() # Uncomment if running locally

if __name__ == "__main__":
    plot_activations()
    
    # Practical PyTorch implementations
    print("\n--- PyTorch Softmax Demo ---")
    logits = torch.tensor([2.0, 1.0, 0.1])
    probs = torch.softmax(logits, dim=0)
    print(f"Logits: {logits}")
    print(f"Probabilities: {probs}")
    print(f"Sum: {probs.sum().item()}")
