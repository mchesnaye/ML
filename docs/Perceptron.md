# 🧠 The Perceptron

The perceptron is one of the earliest and most fundamental building blocks of modern machine learning algorithms. It was introduced in 1958 by Frank Rosenblatt and was inspired by how neurons fire in the brain. It was applied by Rosenblatt in a visual pattern recognition task where it learned to recognize simple black and white shapes (e.g. squares or triangles) within a 20x20 grid of light-sensitive cells (400 pixesl). 

---

## The algorithm

The Perceptron is a simple supervised learning algorithm for binary classification tasks. It attempts to find a linear decision boundary that separates two classes by learning a set of parameters (denoted by weights) using training data with pre-assigned class labels - hence supervised learning. The input to the Perceptronis an n-dimensional vector: 

x = [x₁, x₂, ..., xₙ]

where each input is connected to the Perceptron via a weight. The vector of weights is thus also n-dimensional: 


w = [w₁, w₂, ..., wₙ]

with an additional weight, called the bias, not being linked to any inputs. Next, a weighted sum is computed:

s = w · x + b

which is passed through an activation function to arrive at a prediction. A commonly used activation function is the step function: 

ŷ = 1 if s > 0 else 0

which assigns label 0 (class 1) if s is negative, else it assigns label 1 (class 2). 

Learning in the Perceptron involves iteratively updating the weights based on the error. Specifically, the error is computed as ε = y - ŷ, and the are adjusted using:

wᵢ ← wᵢ + η × ε × xᵢ
b ← b + η × ε

where b is the bias and η is the learning rate. 

---

## 🚀 Python implementation

Classification Problem: We generate two classes of data. Class 1 is represented by 2-dimensional features, randomly sampled from a normal and a uniform distribution. Class 2 is represented by 2-dimensional features, randomly sampled from an exponential and chi-squared distribution. Both classes are arbitrarily offset by a value of 3. Training data comprises 1000 samples from each class. Note that for this toy example, classes are not perfectly linearly seperable. 

![Figure 1: ](images/Figure1.png)


---

## 📚 References

Rosenblatt, F. (1958). The perceptron: A probabilistic model for information storage and organization in the brain. *Psychological Review, 65*(6), 386–408. https://doi.org/10.1037/h0042519

