import numpy as np
import torch as th
import matplotlib.pyplot as plt
import random
# from test import Perceptron

# Define the Perceptron class
class Perceptron:
    def __init__(self, input_size, learning_rate=0.01, epochs=100):
        #self.weights = np.zeros(input_size + 1)  # +1 for bias
        self.weights = np.ones(input_size + 1)  # +1 for bias
        self.learning_rate = learning_rate
        self.epochs = epochs

    def activate(self, x):
        return 1 if x >= 0 else 0
    
    def predict(self, x):
        s = np.dot(self.weights[1:], x) + self.weights[0] 
        a = self.activate(s)
        return a
    
    def train(self, D, L):
        n_samples = D.shape[0]
        ErrSum = []
        for _ in range(self.epochs):

            total_error = 0
            indices = np.random.permutation(n_samples)
            D_shuffled = D[indices]
            L_shuffled = L[indices]

            for i in range(n_samples):
                x = D_shuffled[i]
                error = L_shuffled[i] - self.predict(x)
                self.weights[0] += self.learning_rate * error
                self.weights[1:] += self.learning_rate * error * x
                total_error += abs(error)
            ErrSum.append(total_error)
            print(total_error)
        return ErrSum
            

# Generate training data
random.seed(42)
TSize = 1000    
V1 = np.random.normal(0, 1, TSize)
V2 = np.random.uniform(-1, 1, TSize)
D1 = np.column_stack((V1, V2))
L1 = np.zeros(TSize)

V3 = np.random.exponential(1, TSize)
V4 = np.random.chisquare(2, TSize)    
D2 = np.column_stack((V3, V4))
L2 = np.ones(TSize)

D = np.vstack((D1, D2))
L = np.hstack((L1, L2))


if __name__ == "__main__":
    # Train the perceptron
    P = Perceptron(input_size=2, learning_rate=0.0001, epochs=500)
    ErrSum = P.train(D, L)

    # Test the perceptron
    print("Final Weights:", P.weights)
    correct = 0
    for i in range(len(D)):
        prediction = P.predict(D[i])
        if prediction == L[i]:
            correct += 1
    accuracy = correct / len(D) * 100
    print(f"Accuracy: {accuracy:.2f}%")

    # Plot the training error (ErrSum) over epochs
    plt.figure(figsize=(8, 6))
    plt.plot(range(P.epochs), ErrSum, label='Total Error (ErrSum)', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Total Error')
    plt.title('Training Error Over Epochs')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Plot the scatter plot with the decision boundary
    plt.figure(figsize=(8, 6))
    plt.scatter(D1[:, 0], D1[:, 1], color='blue', label='Population 1 (Label 0)', alpha=0.6)
    plt.scatter(D2[:, 0], D2[:, 1], color='red', label='Population 2 (Label 1)', alpha=0.6)

    # Compute the decision boundary
    w0, w1, w2 = P.weights  # Bias, weight for Feature 1, weight for Feature 2
    x1_min, x1_max = D[:, 0].min() - 0.5, D[:, 0].max() + 0.5  # Range for Feature 1
    x1 = np.array([x1_min, x1_max])
    if w2 != 0:  # Avoid division by zero
        x2 = -(w0 + w1 * x1) / w2  # Compute x2 for the decision boundary
        plt.plot(x1, x2, 'k-', label='Decision Boundary')
    else:
        print("Warning: w2 is zero, cannot plot decision boundary as a line.")

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Scatter Plot with Perceptron Decision Boundary')
    plt.legend()
    plt.grid(True)
    plt.show()

    