# 🧠 The Perceptron

The perceptron is one of the earliest and most fundamental building blocks of modern machine learning algorithms. It was introduced in 1958 by Frank Rosenblatt and was inspired by how neurons fire in the brain. It was applied by Rosenblatt in a visual pattern recognition task where it learned to recognize simple black and white shapes (e.g. squares or triangles) within a 20x20 grid of light-sensitive cells (400 pixesl). 

---

## How the Perceptron Works

The Perceptron is a simple learning algorithm used for binary classification tasks. It attempts to find a linear decision boundary that separates the two classes by learning an n-dimensional set of weights:

w = [w₁, w₂, ..., wₙ]

along with a bias term b. Each weight is linked to an input feature, which is also n-dimensional:

x = [x₁, x₂, ..., xₙ]

Next, a weighted sum is computed:

s = w · x + b

To arrive at an output, the s is then passed through an activation function, such as a step function: 

ŷ = 1 if s > 0 else 0

which thus assigns label 0 (class 1) is s is negative, else label 1 (class 2) is assigned. 

The perceptron learns by iteratively updating the weights based on the error, computed as ε = y - ŷ. The weights are then adjusted as:

wᵢ ← wᵢ + η × ε × xᵢ
b ← b + η × ε

where η is the learning rate. 

---

## 🚀 What is a Perceptron?

A *perceptron* is one of the simplest types of artificial neural networks, used for binary classification tasks. It learns a linear decision boundary between two classes by adjusting weights based on classification errors.

---



## 📚 References

Rosenblatt, F. (1958). The perceptron: A probabilistic model for information storage and organization in the brain. *Psychological Review, 65*(6), 386–408. https://doi.org/10.1037/h0042519

