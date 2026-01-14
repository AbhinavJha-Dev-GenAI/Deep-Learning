import torch
import torch.nn as nn
from torchvision import models

def transfer_learning_demo():
    # 1. Feature Extraction Mode
    print("--- Transfer Learning: Feature Extraction ---")
    model_fe = models.resnet18(pretrained=True)
    
    # Freeze ALL parameters
    for param in model_fe.parameters():
        param.requires_grad = False
        
    # Replace the classification head (Last layer)
    num_ftrs = model_fe.fc.in_features
    model_fe.fc = nn.Linear(num_ftrs, 2) # e.g. Binary classification
    
    # Verify grad status
    trainable_params = sum(p.numel() for p in model_fe.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model_fe.parameters())
    print(f"Trainable Parameters (FE): {trainable_params}")
    print(f"Total Parameters: {total_params}")

    # 2. Fine-Tuning Mode
    print("\n--- Transfer Learning: Fine-Tuning ---")
    model_ft = models.resnet18(pretrained=True)
    
    # Unfreeze ALL or SOME parameters
    for param in model_ft.parameters():
        param.requires_grad = True # Everything is trainable
        
    # Replace head
    model_ft.fc = nn.Linear(num_ftrs, 2)
    
    trainable_params_ft = sum(p.numel() for p in model_ft.parameters() if p.requires_grad)
    print(f"Trainable Parameters (FT): {trainable_params_ft}")

    # 3. Learning Rate Strategy Note
    print("\nNote: For Fine-Tuning, always use a significantly SMALLER learning rate (e.g., 1e-5)")
    print("to preserve pre-trained features.")

if __name__ == "__main__":
    transfer_learning_demo()
