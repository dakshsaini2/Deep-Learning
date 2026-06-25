# Day 41 - Backpropagation in Convolutional Neural Networks (CNNs)

## Introduction

In Artificial Neural Networks (ANNs), backpropagation updates the weights of fully connected layers by propagating the error backward.

Convolutional Neural Networks (CNNs) also use backpropagation, but the process is more complex because CNNs contain:

- Convolution Layers
- Activation Layers
- Pooling Layers
- Fully Connected Layers

The goal remains the same:

```text
Minimize the Loss Function
by Updating Network Parameters
```

Backpropagation enables CNNs to automatically learn filters that detect edges, textures, shapes, and objects.

Without backpropagation:

```text
CNNs Cannot Learn
```

---

# What is Backpropagation?

Backpropagation is an optimization algorithm that computes:

```text
Gradient of Loss
with Respect to Every Parameter
```

These gradients are then used to update:

- Filters (Kernels)
- Biases
- Weights

using Gradient Descent or another optimizer.

---

# CNN Training Pipeline

```text
Input Image
      ↓
Convolution
      ↓
ReLU
      ↓
Pooling
      ↓
Flatten
      ↓
Fully Connected Layer
      ↓
Prediction
      ↓
Loss
      ↓
Backpropagation
      ↓
Weight Update
```

---

# Goal of Backpropagation

Suppose the network predicts:

```text
Dog
```

Actual label:

```text
Cat
```

Error:

```text
High Loss
```

Backpropagation adjusts the parameters so the next prediction is closer to the correct label.

---

# Forward Pass

During the forward pass:

```text
Image
↓
Feature Extraction
↓
Classification
↓
Prediction
```

No learning occurs yet.

---

# Loss Calculation

The prediction is compared with the true label.

Example:

Prediction:

```text
0.72
```

Actual:

```text
1.00
```

Loss is computed using a loss function such as:

- Cross Entropy Loss
- Mean Squared Error (MSE)

---

# Backward Pass

The error is propagated backward through the network.

```text
Loss
 ↓
Output Layer
 ↓
Dense Layer
 ↓
Flatten
 ↓
Pooling
 ↓
ReLU
 ↓
Convolution
```

Gradients are calculated at every step.

---

# Chain Rule

Backpropagation uses the:

```text
Chain Rule of Calculus
```

to compute gradients efficiently.

If the loss depends on multiple functions, the derivative is calculated by multiplying intermediate derivatives.

---

# Weight Update Rule

The parameters are updated using:



Where:

- W = Weight or Filter Value
- η = Learning Rate
- L = Loss Function

---

# Backpropagation Through CNN Layers

The error passes through every layer in reverse order.

```text
Output Layer
       ↑
Dense Layer
       ↑
Flatten Layer
       ↑
Pooling Layer
       ↑
ReLU Layer
       ↑
Convolution Layer
```

---

# Backpropagation in Fully Connected Layer

The process is the same as in ANN.

Steps:

1. Compute gradient.
2. Update weights.
3. Pass error backward.

---

# Backpropagation in Flatten Layer

Flatten has:

```text
No Trainable Parameters
```

It simply reshapes:

```text
1D Vector
```

back into:

```text
3D Feature Maps
```

during the backward pass.

---

# Backpropagation in Pooling Layer

Pooling layers usually have:

```text
No Learnable Parameters
```

Only gradients are propagated.

---

# Max Pooling Backpropagation

Example:

Forward Pass:

```text
2 6

5 3
```

Output:

```text
6
```

During backpropagation:

Gradient is assigned only to:

```text
6
```

Other values receive:

```text
0
```

---

# Average Pooling Backpropagation

Suppose:

```text
2×2 Pool
```

Gradient:

```text
8
```

Backward:

```text
2 2

2 2
```

Gradient is distributed equally among all inputs.

---

# Backpropagation Through ReLU

ReLU Function:



Derivative:



If:

```text
Input > 0
```

Gradient passes unchanged.

If:

```text
Input ≤ 0
```

Gradient becomes zero.

---

# Backpropagation Through Convolution Layer

This is the most important step.

CNN learns by updating:

```text
Filters (Kernels)
```

During backpropagation:

- Compute gradient for each filter.
- Compute gradient for each bias.
- Update filter values.

The filter gradually learns useful features such as:

```text
Edges
Corners
Textures
```

---

# Filter Update

Example:

Initial Filter:

```text
0.1 0.3 -0.2

0.2 -0.1 0.5

0.0 0.4 -0.3
```

After many iterations:

```text
Edge Detector
```

The filter becomes specialized for detecting meaningful patterns.

---

# Gradient Flow

```text
Loss
 ↓
Dense Layer
 ↓
Flatten
 ↓
Pooling
 ↓
ReLU
 ↓
Convolution
```

Each layer computes:

```text
Gradient
```

and passes it to the previous layer.

---

# Complete CNN Training Cycle

```text
Input Image
      ↓
Forward Pass
      ↓
Prediction
      ↓
Loss Function
      ↓
Backpropagation
      ↓
Gradient Calculation
      ↓
Optimizer
      ↓
Weight Update
      ↓
Repeat
```

---

# Role of Optimizers

Backpropagation computes:

```text
Gradients
```

Optimizers decide:

```text
How Parameters Are Updated
```

Common optimizers:

- SGD
- Momentum
- RMSProp
- Adam
- AdamW

---

# Challenges During Backpropagation

## Vanishing Gradient

Gradients become extremely small.

Result:

```text
Slow Learning
```

---

## Exploding Gradient

Gradients become very large.

Result:

```text
Unstable Training
```

---

# Solutions

Modern CNNs use:

- ReLU Activation
- Batch Normalization
- He Initialization
- Residual Connections
- Adam Optimizer

to improve gradient flow.

---

# TensorFlow Example

```python
import tensorflow as tf

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=32
)
```

TensorFlow automatically performs:

- Forward Propagation
- Loss Computation
- Backpropagation
- Weight Updates

---

# PyTorch Example

```python
loss = criterion(output, target)

optimizer.zero_grad()

loss.backward()

optimizer.step()
```

`loss.backward()` computes gradients, and `optimizer.step()` updates the parameters.

---

# Real World Applications

Backpropagation is used in:

- Image Classification
- Face Recognition
- Medical Diagnosis
- Autonomous Vehicles
- Satellite Image Analysis
- Object Detection

Every trainable CNN relies on backpropagation.

---

# Important Terms

| Term | Meaning |
|---------|---------|
| Backpropagation | Computes gradients by propagating errors backward |
| Gradient | Direction of steepest increase in loss |
| Chain Rule | Mathematical rule used for gradient computation |
| Filter | Learnable convolution kernel |
| Max Pooling | Passes gradient only to the maximum value |
| Optimizer | Updates parameters using gradients |

---

# Interview Questions

## Q1. Why is backpropagation needed in CNN?

To compute gradients and update filters, weights, and biases so that the loss decreases.

---

## Q2. Which CNN layer has trainable parameters?

The Convolution Layer and Fully Connected Layer.

---

## Q3. Does the Pooling Layer have trainable parameters?

No. It only routes gradients during backpropagation.

---

## Q4. What happens during backpropagation through Max Pooling?

Only the input that produced the maximum value receives the gradient.

---

## Q5. Which mathematical concept is the foundation of backpropagation?

The Chain Rule of Calculus.

---

# Key Points Learned Today

✅ Learned Backpropagation in CNNs

✅ Understood Forward and Backward Pass

✅ Learned Backpropagation through Convolution Layers

✅ Understood Backpropagation through ReLU

✅ Learned Max Pooling and Average Pooling Backpropagation

✅ Studied Gradient Flow

✅ Connected Backpropagation with Optimizers

---

# Mini Summary

Backpropagation is the learning mechanism of Convolutional Neural Networks. After the forward pass and loss computation, gradients are propagated backward through the fully connected, flatten, pooling, activation, and convolution layers using the chain rule. These gradients are then used by optimizers such as Adam or SGD to update filters and weights, enabling CNNs to learn increasingly meaningful visual features.

---

# Progress

✅ Completed Day 41 - Backpropagation in CNNs
