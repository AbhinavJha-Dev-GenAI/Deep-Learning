import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(residual) # The Skip Connection
        out = self.relu(out)
        return out

if __name__ == "__main__":
    # Test Output Dimensions
    print("--- CNN Block Dimension Test ---")
    x = torch.randn(1, 64, 32, 32)
    
    # Standard Conv
    conv = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
    print(f"Input: {x.shape}")
    print(f"After Standard Conv (Stride 2): {conv(x).shape}")
    
    # Residual Block
    res_block = ResidualBlock(64, 128, stride=2)
    print(f"After Residual Block (Stride 2): {res_block(x).shape}")
    
    # MaxPool
    pool = nn.MaxPool2d(kernel_size=2, stride=2)
    print(f"After MaxPool (2x2): {pool(x).shape}")
