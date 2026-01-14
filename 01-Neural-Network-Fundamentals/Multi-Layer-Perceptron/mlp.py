import torch
import torch.nn as nn
import torch.optim as optim

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.sigmoid(x)
        return x

if __name__ == "__main__":
    # Solve XOR (Non-linearly separable)
    X = torch.tensor([[0,0], [0,1], [1,0], [1,1]], dtype=torch.float32)
    y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

    model = MLP(2, 4, 1)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.1)

    print("Training MLP to solve XOR...")
    for epoch in range(200):
        # Forward pass
        outputs = model(X)
        loss = criterion(outputs, y)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch+1) % 50 == 0:
            print(f'Epoch [{epoch+1}/200], Loss: {loss.item():.4f}')

    print("\nResults after training:")
    with torch.no_grad():
        results = model(X)
        for i, val in enumerate(results):
            print(f"Input: {X[i].tolist()}, Target: {y[i].item()}, Predicted: {val.item():.4f} -> {int(val > 0.5)}")
    
    # Check if XOR is solved
    final_preds = (results > 0.5).float()
    if torch.equal(final_preds, y):
        print("\nSuccess: XOR problem solved!")
    else:
        print("\nNote: Might need more epochs or a different seed to solve XOR consistently.")
