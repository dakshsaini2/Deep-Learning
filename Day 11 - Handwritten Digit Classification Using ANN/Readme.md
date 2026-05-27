# Day 11 - Handwritten Digit Classification Using ANN
Code Soon  
## Introduction

Handwritten Digit Classification is one of the most famous beginner projects in Deep Learning and Computer Vision.

In this project, an Artificial Neural Network (ANN) learns to recognize handwritten digits from:
- 0 to 9.

This project became popular because it demonstrates:
- image classification,
- neural network learning,
- feature extraction,
- multi-class classification.

The most commonly used dataset is:
- MNIST Dataset.

---

# What is MNIST Dataset?

MNIST stands for:
- Modified National Institute of Standards and Technology.

It contains:
- 70,000 handwritten digit images.

Dataset split:
- 60,000 training images
- 10,000 testing images

---

# Sample MNIST Digits



---

# Why is MNIST Important?

MNIST became the "Hello World" of Deep Learning because it helps beginners understand:
- neural networks,
- image classification,
- training pipelines,
- optimization.

Many Deep Learning breakthroughs were first tested on MNIST.

---

# Problem Statement

Goal:
- predict which digit (0–9) is present in an image.

This is a:
- Multi-Class Classification Problem.

---

# Real World Applications

Handwritten digit recognition is used in:
- Postal Code Recognition
- Bank Check Processing
- OCR Systems
- Form Digitization
- Document Processing

Companies and systems using OCR:
- :contentReference[oaicite:1]{index=1}
- :contentReference[oaicite:2]{index=2}
- :contentReference[oaicite:3]{index=3}

---

# ANN Architecture

```text
Input Layer → Hidden Layer(s) → Output Layer
```

---

# ANN Architecture for MNIST



---

# Understanding the Input

Each MNIST image:
- size = 28 × 28 pixels.

Total pixels:

:contentReference[oaicite:5]{index=5}

The image is flattened into:
- a vector of 784 features.

---

# Input Vector Representation

Input:

:contentReference[oaicite:6]{index=6}

Each value represents:
- pixel intensity.

---

# Pixel Intensity

Pixel values range between:
- 0 → black
- 255 → white

Usually normalized between:
- 0 and 1.

Normalization improves:
- training stability,
- convergence speed.

---

# Why Flatten Images?

ANNs accept:
- vectors as input.

Therefore:
- 2D image → 1D vector conversion.

Example:

```text
28×28 Image → 784-Dimensional Vector
```

---

# Hidden Layers

Hidden layers learn:
- edges,
- curves,
- digit patterns,
- shapes.

As depth increases:
- feature understanding improves.

---

# Output Layer

Since there are:
- 10 classes (0–9),

output layer contains:
- 10 neurons.

Each neuron predicts probability for one digit.

---

# Softmax Activation

Output layer uses:
- Softmax Function.

Softmax converts outputs into:
- probability distribution.

Formula:

:contentReference[oaicite:7]{index=7}

---

# Why Softmax?

Softmax ensures:
- all probabilities sum to 1,
- highest probability becomes prediction.

Example:

```text
[0.01, 0.02, 0.90, 0.03, ...]
```

Prediction:
- Digit = 2

---

# Forward Propagation

Each layer computes:

:contentReference[oaicite:8]{index=8}

Activation:

:contentReference[oaicite:9]{index=9}

---

# Activation Functions

## Hidden Layers → ReLU

:contentReference[oaicite:10]{index=10}

Advantages:
- fast,
- efficient,
- reduces vanishing gradients.

---

# Output Layer → Softmax

Used for:
- multi-class classification.

---

# Loss Function

Since MNIST is multi-class classification:

We use:
- Categorical Cross Entropy Loss.

Formula:

:contentReference[oaicite:11]{index=11}

Where:
- `yᵢ` = actual label,
- `ŷᵢ` = predicted probability.

---

# Training Process

Training steps:
1. Forward Propagation
2. Loss Calculation
3. Backpropagation
4. Weight Updates

Repeated across:
- multiple epochs.

---

# Backpropagation

Backpropagation computes:
- gradients,
- weight updates,
- optimization direction.

Modern Deep Learning depends heavily on:
- gradient descent,
- automatic differentiation.

---

# Why ANN Learns Digits Successfully?

ANNs learn:
- hidden visual patterns,
- pixel relationships,
- feature hierarchies.

Example:
- curves,
- loops,
- strokes,
- digit structures.

---

# Visualization of Learning

```text
Raw Pixels
   ↓
Edges
   ↓
Shapes
   ↓
Digit Representation
   ↓
Prediction
```

---

# Data Preprocessing

Common preprocessing steps:

## 1. Normalization

Scale pixel values:

```python
X = X / 255.0
```

---

# 2. Flattening

Convert:
- 28×28 → 784 vector.

---

# 3. One-Hot Encoding

Digit labels converted into vectors.

Example:

```text
Digit 3 → [0,0,0,1,0,0,0,0,0,0]
```

---

# Evaluation Metrics

## Accuracy

Measures:
- percentage of correct predictions.

MNIST models often achieve:
- 95%–99% accuracy.

---

# Confusion Matrix

Shows:
- which digits are confused with others.

Example:
- 5 mistaken as 3.

---

# Overfitting Problem

Large neural networks may:
- memorize training images.

Solutions:
- Dropout
- Regularization
- Early Stopping

---

# Why CNNs Became Better?

ANNs flatten images and lose:
- spatial information.

CNNs preserve:
- local image structure,
- spatial hierarchies.

Therefore CNNs outperform traditional ANNs in Computer Vision.

---

# ANN vs CNN for Image Tasks

| ANN | CNN |
|---|---|
| Flattened input | Preserves spatial structure |
| Less efficient for images | Specialized for vision |
| More parameters | Parameter sharing |
| Lower image performance | Higher accuracy |

---

# Industry Perspective

Handwritten digit classification introduced core ideas used in:
- OCR,
- Computer Vision,
- Autonomous Systems,
- Medical Imaging,
- Face Recognition.

Modern AI systems evolved from these foundations.

---

# TensorFlow/Keras Implementation

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist

# Load Dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize
X_train = X_train / 255.0
X_test = X_test / 255.0

# Build ANN
model = Sequential([
    Flatten(input_shape=(28,28)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile Model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train Model
model.fit(X_train, y_train, epochs=5)

# Evaluate
model.evaluate(X_test, y_test)
```

---

# Why Adam Optimizer?

Adam combines:
- Momentum,
- Adaptive Learning Rates.

Advantages:
- fast convergence,
- stable optimization,
- excellent practical performance.

---

# Important Terms

| Term | Meaning |
|---|---|
| MNIST | Handwritten digit dataset |
| Softmax | Multi-class probability activation |
| Flattening | Converting image into vector |
| Cross Entropy | Multi-class loss function |
| Epoch | One complete training cycle |
| OCR | Optical Character Recognition |

---

# Interview Questions

## Q1. Why is Softmax used in MNIST?
Because digit classification is a multi-class problem.

## Q2. Why normalize images?
To improve training stability and convergence.

## Q3. Why are CNNs better for images?
Because they preserve spatial information.

## Q4. Why flatten images in ANN?
Because fully connected networks require vector inputs.

---

# Key Points Learned Today

✅ Learned handwritten digit classification  
✅ Understood MNIST dataset  
✅ Learned ANN image classification pipeline  
✅ Understood Softmax activation  
✅ Learned categorical cross entropy loss  
✅ Understood flattening and normalization  
✅ Learned ANN limitations in Computer Vision

---

# Mini Summary

Handwritten digit classification using ANN is one of the most foundational Deep Learning projects. It demonstrates how neural networks learn visual patterns from pixel data to perform multi-class image classification. This project introduced key ideas that later evolved into modern Computer Vision systems.

---
#DakshSaini
# Progress

✅ Completed Day 11 - Handwritten Digit Classification Using ANN
