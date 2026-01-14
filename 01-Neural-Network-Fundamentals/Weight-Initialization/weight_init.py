import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def analyze_activations(init_type, activation_fn):
    # Deep network to see signal propagation
    layers = []
    input_dim = 512
    for _ in range(50):
        layer = nn.Linear(input_dim, input_dim)
        
        # Initialize
        if init_type == "zero":
            nn.init.zeros_(layer.weight)
        elif init_type == "normal_large":
            nn.init.normal_(layer.weight, mean=0, std=1)
        elif init_type == "xavier":
            nn.init.xavier_normal_(layer.weight)
        elif init_type == "kaiming":
            nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
            
        layers.append(layer)
        layers.append(activation_fn)

    model = nn.Sequential(*layers)
    
    # Input
    x = torch.randn(1, input_dim)
    
    # Track mean and std across layers
    means = []
    stds = []
    
    current_x = x
    for layer in model:
        current_x = layer(current_x)
        if isinstance(layer, nn.Linear):
            means.append(current_x.mean().item())
            stds.append(current_x.std().item())
            
    return means, stds

def plot_init_comparison():
    plt.figure(figsize=(12, 5))
    
    # Test different combinations
    # 1. Sigmoid with Xavier (Correct) vs Normal (Often bad)
    m1, s1 = analyze_activations("xavier", nn.Sigmoid())
    m2, s2 = analyze_activations("normal_large", nn.Sigmoid())
    
    plt.subplot(1, 2, 1)
    plt.plot(s1, label='Xavier + Sigmoid (Stable)')
    plt.plot(s2, label='Normal (Scale 1) + Sigmoid (Vanishing)')
    plt.title("Standard Deviation of Activations (Sigmoid)")
    plt.xlabel("Layer Depth")
    plt.ylabel("Std Dev")
    plt.legend()
    plt.grid(True)

    # 2. ReLU with Kaiming (Correct) vs Xavier (Slightly worse)
    m3, s3 = analyze_activations("kaiming", nn.ReLU())
    m4, s4 = analyze_activations("xavier", nn.ReLU())
    
    plt.subplot(1, 2, 2)
    plt.plot(s3, label='Kaiming + ReLU (Stable)')
    plt.plot(s4, label='Xavier + ReLU (Fading)')
    plt.title("Standard Deviation of Activations (ReLU)")
    plt.xlabel("Layer Depth")
    plt.ylabel("Std Dev")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    save_path = "weight_init_comparison.png"
    plt.savefig(save_path)
    print(f"Initialization plot saved to {save_path}")

if __name__ == "__main__":
    plot_init_comparison()
    
    # Show basic usage
    print("\n--- Basic Weight Initialization ---")
    layer = nn.Linear(10, 5)
    nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
    print("Layer weights initialized with He/Kaiming Normal.")
