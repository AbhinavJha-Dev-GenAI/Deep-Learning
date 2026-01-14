import numpy as np
import torch
import torch.nn as nn

# 1. Manual Implementation using NumPy
class ManualPerceptron:
    def __init__(self, input_size, lr=0.01, epochs=100):
        self.weights = np.zeros(input_size + 1) # +1 for bias
        self.lr = lr
        self.epochs = epochs

    def activation(self, x):
        return 1 if x >= 0 else 0

    def predict(self, x):
        z = self.weights.T.dot(np.append(x, 1)) # Append 1 for bias
        return self.activation(z)

    def train(self, X, y):
        for _ in range(self.epochs):
            for i in range(X.shape[0]):
                y_pred = self.predict(X[i])
                error = y[i] - y_pred
                self.weights += self.lr * error * np.append(X[i], 1)

# 2. PyTorch Implementation
class PyTorchPerceptron(nn.Module):
    def __init__(self, input_size):
        super(PyTorchPerceptron, self).__init__()
        self.fc = nn.Linear(input_size, 1)
        
    def forward(self, x):
        return torch.where(self.fc(x) > 0, 1.0, 0.0)

if __name__ == "__main__":
    # Logic Gate: AND
    X = np.array([[0,0], [0,1], [1,0], [1,1]])
    y = np.array([0, 0, 0, 1])

    print("--- Manual Perceptron (AND Gate) ---")
    p = ManualPerceptron(input_size=2)
    p.train(X, y)
    for x_test in X:
        print(f"Input: {x_test}, Predicted: {p.predict(x_test)}")

    print("\n--- PyTorch Perceptron (Inference) ---")
    pt_p = PyTorchPerceptron(input_size=2)
    test_tensor = torch.tensor([[0.5, 0.5]])
    print(f"Input: {test_tensor.tolist()}, Predicted: {pt_p(test_tensor).item()}")
