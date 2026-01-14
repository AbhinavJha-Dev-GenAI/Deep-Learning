import numpy as np

def perceptron_from_scratch(X, y, lr=0.1, epochs=10):
    """
    Challenge: Implement a basic Perceptron using only NumPy.
    """
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    bias = 0
    
    for _ in range(epochs):
        for idx, x_i in enumerate(X):
            # linear model
            linear_output = np.dot(x_i, weights) + bias
            # activation
            y_predicted = 1 if linear_output >= 0 else 0
            
            # Perceptron update rule
            update = lr * (y[idx] - y_predicted)
            weights += update * x_i
            bias += update
            
    return weights, bias

def calculate_conv_output_shape(W, K, P, S):
    """
    Challenge: Calculate the output dimension of a convolution layer.
    Formula: O = (W - K + 2P) / S + 1
    
    Args:
        W: Input width/height
        K: Kernel size
        P: Padding
        S: Stride
    """
    output = ((W - K + 2 * P) / S) + 1
    return int(output)

if __name__ == "__main__":
    print("--- Interview Challenge: NumPy Perceptron (OR Gate) ---")
    X = np.array([[0,0], [0,1], [1,0], [1,1]])
    y = np.array([0, 1, 1, 1])
    
    w, b = perceptron_from_scratch(X, y)
    print(f"Learned Weights: {w}, Bias: {b}")
    
    # Test
    for x_i in X:
        pred = 1 if np.dot(x_i, w) + b >= 0 else 0
        print(f"Input: {x_i}, Predicted: {pred}")

    print("\n--- Interview Challenge: Conv Shape Calculator ---")
    # Example: 224x224 input, 3x3 kernel, padding 1, stride 2
    res = calculate_conv_output_shape(W=224, K=3, P=1, S=2)
    print(f"Input 224x224, K=3, P=1, S=2 -> Output Shape: {res}x{res}")
    
    # Standard 3x3 Conv (Same Padding)
    res_same = calculate_conv_output_shape(W=224, K=3, P=1, S=1)
    print(f"Input 224x224, K=3, P=1, S=1 -> Output Shape: {res_same}x{res_same} (Same Padding)")
