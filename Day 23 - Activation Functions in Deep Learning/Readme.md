# Day 23 - Activation Functions in Deep Learning

## Introduction

Activation Functions are one of the most important building blocks of Neural Networks.

Without activation functions:

```text
Deep Neural Networks
=
Simple Linear Models
```

No matter how many layers are added, the network would only learn linear relationships.

Activation Functions introduce:

- Non-Linearity
- Complex Pattern Learning
- Feature Extraction
- Decision Boundaries

Modern AI systems like:

- ChatGPT
- Gemini
- Computer Vision Models
- Recommendation Systems

all rely on activation functions.

---

# What is an Activation Function?

An Activation Function is a mathematical function applied to the output of a neuron.

Neural Network Computation:



Activation Output:



Where:

- Z = Weighted Sum
- g(Z) = Activation Function
- A = Final Output

---

# Why Do We Need Activation Functions?

Without activation functions:

```text
Input
 ↓
Linear Layer
 ↓
Linear Layer
 ↓
Linear Layer
 ↓
Output
```

The entire network becomes:

```text
One Big Linear Equation
```

which limits learning ability.

---

# Benefits of Activation Functions

✅ Introduce Non-Linearity

✅ Learn Complex Patterns

✅ Enable Deep Learning

✅ Improve Feature Extraction

✅ Create Decision Boundaries

---

# Types of Activation Functions

Major activation functions:

1. Binary Step Function
2. Linear Activation
3. Sigmoid
4. Tanh
5. ReLU
6. Leaky ReLU
7. ELU
8. Softmax
9. GELU

---

# 1. Binary Step Function

One of the earliest activation functions.

Formula:

```text
Output = 1 if x ≥ 0
Output = 0 if x < 0
```

---

# Binary Step Visualization

```text
      1 ────────
      |
      |
------|---------
      |
      |
      0
```

---

# Limitations

❌ Not Differentiable

❌ Cannot Use Backpropagation

❌ Not Used in Modern Deep Learning

---

# 2. Linear Activation Function

Formula:



Output:

```text
Input = Output
```

---

# Problem with Linear Activation

Even with:

```text
100 Layers
```

network behaves like:

```text
Single Linear Layer
```

Not suitable for hidden layers.

---

# 3. Sigmoid Activation Function

One of the most famous activation functions.

Formula:



Range:

```text
0 to 1
```

---

# Sigmoid Properties

Advantages:

✅ Smooth Curve

✅ Probability Output

✅ Useful for Binary Classification

---

# Sigmoid Problems

❌ Vanishing Gradient Problem

❌ Slow Training

❌ Not Zero Centered

---

# Applications

Mostly used in:

```text
Output Layer
```

for Binary Classification.

---

# 4. Tanh Activation Function

Formula:



Range:

```text
-1 to 1
```

---

# Advantages of Tanh

✅ Zero-Centered

✅ Better than Sigmoid

✅ Stronger Gradients

---

# Limitations

❌ Still suffers from Vanishing Gradients

---

# 5. ReLU (Rectified Linear Unit)

Most important activation function in Deep Learning.

Formula:



---

# ReLU Behavior

```text
x < 0 → 0

x > 0 → x
```

---

# Why ReLU Became Popular?

Advantages:

✅ Computationally Efficient

✅ Fast Training

✅ Reduces Vanishing Gradients

✅ Enables Deep Networks

---

# ReLU Visualization

```text
        /
       /
      /
-----/
```

---

# ReLU Limitation

Problem:

```text
Dying ReLU
```

If neuron always receives negative values:

```text
Output = 0
```

Neuron stops learning.

---

# 6. Leaky ReLU

Solution to Dying ReLU.

Formula:



---

# Advantages

✅ Solves Dying ReLU

✅ Maintains Gradient Flow

✅ Better Learning

---

# 7. ELU (Exponential Linear Unit)

Formula:

```text
x                if x > 0
α(e^x - 1)       if x ≤ 0
```

---

# Benefits

✅ Faster Convergence

✅ Better Gradient Flow

✅ Handles Negative Values Better

---

# 8. Softmax Activation Function

Used for:

```text
Multi-Class Classification
```

Formula:



---

# Example

Output:

```text
[0.1, 0.2, 0.7]
```

Interpretation:

```text
Class 3 = 70% Probability
```

---

# Applications

Used in:

- MNIST Classification
- Image Recognition
- NLP Classification

---

# 9. GELU (Gaussian Error Linear Unit)

Modern activation used in:

- Transformers
- BERT
- GPT Models

Formula:

```text
Approximation:
0.5x(1+tanh(...))
```

---

# Why GELU?

Advantages:

✅ Smooth Activation

✅ Better Performance

✅ State-of-the-Art Results

---

# Activation Functions Comparison

| Function | Range | Common Usage |
|-----------|--------|-------------|
| Step | 0,1 | Historical |
| Linear | (-∞,∞) | Regression Output |
| Sigmoid | (0,1) | Binary Classification |
| Tanh | (-1,1) | Older Networks |
| ReLU | [0,∞) | Hidden Layers |
| Leaky ReLU | (-∞,∞) | Hidden Layers |
| Softmax | (0,1) | Multi-Class Output |
| GELU | (-∞,∞) | Transformers |

---

# Choosing Activation Functions

### Hidden Layers

Recommended:

```text
ReLU
Leaky ReLU
GELU
```

---

### Binary Classification

Output Layer:

```text
Sigmoid
```

---

### Multi-Class Classification

Output Layer:

```text
Softmax
```

---

### Regression

Output Layer:

```text
Linear Activation
```

---

# Industry Best Practices

Modern Deep Learning commonly uses:

```text
Hidden Layers → ReLU/GELU

Binary Output → Sigmoid

Multi-Class Output → Softmax
```

---

# TensorFlow Example

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()

model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))
```

---

# Real World Applications

Activation Functions are used in:

- Computer Vision
- NLP
- Recommendation Systems
- Speech Recognition
- Healthcare AI
- Autonomous Vehicles

Every Deep Learning model depends on them.

---

# Important Terms

| Term | Meaning |
|---------|---------|
| Activation Function | Adds non-linearity |
| ReLU | Most popular hidden layer activation |
| Sigmoid | Binary classification activation |
| Softmax | Multi-class classification activation |
| GELU | Transformer activation |
| Dying ReLU | ReLU neuron stops learning |

---

# Interview Questions

## Q1. Why are activation functions needed?

They introduce non-linearity, allowing neural networks to learn complex patterns.

---

## Q2. Why is ReLU popular?

Because it is simple, fast, and reduces vanishing gradients.

---

## Q3. What activation is used for binary classification?

Sigmoid.

---

## Q4. What activation is used for multi-class classification?

Softmax.

---

## Q5. What activation is used in GPT and Transformers?

GELU.

---

# Key Points Learned Today

✅ Learned Activation Functions

✅ Understood Non-Linearity

✅ Learned Sigmoid and Tanh

✅ Understood ReLU and Leaky ReLU

✅ Learned Softmax

✅ Studied GELU and Transformer Activations

✅ Learned Industry Best Practices

---

# Mini Summary

Activation Functions are mathematical functions that introduce non-linearity into neural networks, enabling them to learn complex patterns. ReLU dominates modern Deep Learning, while Sigmoid, Softmax, and GELU are used for specific tasks such as classification and Transformer architectures.

---
#d
# Progress

✅ Completed Day 23 - Activation Functions in Deep Learning
