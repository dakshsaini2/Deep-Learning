# Day 28 - Optimizers in Deep Learning

## Introduction

Training a neural network involves finding the best values for weights and biases that minimize the loss function.

Gradient Descent helps us move toward the minimum loss, but basic Gradient Descent is often:

- Slow
- Computationally Expensive
- Unstable
- Sensitive to Learning Rate

To overcome these limitations, Deep Learning uses:

```text
Optimizers
```

Optimizers are algorithms that improve and accelerate the learning process.

Modern AI systems such as:

- ChatGPT
- Gemini
- Claude
- Stable Diffusion
- Computer Vision Models

all rely on advanced optimizers.

---

# What is an Optimizer?

An Optimizer is an algorithm that:

```text
Updates Weights
to Minimize Loss
```

during training.

General update rule:



Where:

- W = Weight
- η = Learning Rate
- ∂L/∂W = Gradient

---

# Why Do We Need Optimizers?

Basic Gradient Descent faces problems:

❌ Slow Convergence

❌ Gets Stuck in Local Minima

❌ Oscillations

❌ Sensitive Learning Rate

Optimizers help solve these issues.

---

# Evolution of Optimizers

```text
Gradient Descent
       ↓
Stochastic Gradient Descent (SGD)
       ↓
Momentum
       ↓
AdaGrad
       ↓
RMSProp
       ↓
Adam
       ↓
AdamW
```

---

# 1. Gradient Descent

Basic optimizer.

Process:

```text
Compute Gradient
       ↓
Update Weight
       ↓
Repeat
```

Update Rule:



---

# Limitation

Training becomes slow for large datasets.

---

# 2. Stochastic Gradient Descent (SGD)

Instead of using the entire dataset:

```text
One Sample
```

is used to update weights.

Benefits:

✅ Faster Learning

✅ Lower Memory Usage

---

# Limitation

Updates are noisy.

Optimization path becomes unstable.

---

# 3. Momentum Optimizer

Inspired by physics.

Idea:

```text
Keep Moving
in Previous Direction
```

instead of starting fresh every update.

---

# Momentum Formula

Velocity:



Weight Update:



---

# Why Momentum Helps?

Without Momentum:

```text
Zig-Zag Movement
```

With Momentum:

```text
Smooth Movement
```

toward optimum.

---

# Benefits

✅ Faster Convergence

✅ Less Oscillation

✅ Better Stability

---

# 4. AdaGrad

Full Form:

```text
Adaptive Gradient Algorithm
```

Main Idea:

```text
Different Learning Rate
for Every Parameter
```

---

# Advantages

✅ Good for Sparse Data

✅ Automatic Learning Rate Adjustment

---

# Limitation

Learning rate continuously decreases.

Eventually:

```text
Learning Almost Stops
```

---

# 5. RMSProp

Introduced to fix AdaGrad's problem.

Main Idea:

```text
Use Moving Average
of Squared Gradients
```

instead of accumulating forever.

---

# RMSProp Formula

Running Average:



Update Rule:



---

# Benefits

✅ Faster Training

✅ Stable Learning

✅ Works Well in Deep Networks

---

# 6. Adam Optimizer

Full Form:

```text
Adaptive Moment Estimation
```

Most popular optimizer in Deep Learning.

Adam combines:

- Momentum
- RMSProp

---

# Adam Intuition

Momentum tracks:

```text
Direction
```

RMSProp tracks:

```text
Learning Rate
```

Adam combines both.

---

# Adam Formula

First Moment:



Second Moment:



Weight Update:



---

# Why Adam Became Popular?

Advantages:

✅ Fast Convergence

✅ Adaptive Learning Rates

✅ Works Well on Most Problems

✅ Minimal Hyperparameter Tuning

---

# Default Adam Parameters

Typical values:

```text
Learning Rate = 0.001

β1 = 0.9

β2 = 0.999
```

---

# 7. AdamW

Improved version of Adam.

Main improvement:

```text
Better Weight Decay
```

handling.

---

# Why AdamW?

Traditional Adam sometimes:

```text
Over-Regularizes
```

AdamW separates:

```text
Weight Decay
```

from gradient updates.

---

# Benefits of AdamW

✅ Better Generalization

✅ Improved Performance

✅ Widely Used in Transformers

---

# Optimizer Comparison

| Optimizer | Speed | Stability | Popularity |
|------------|--------|-----------|------------|
| Gradient Descent | Low | High | Low |
| SGD | Medium | Medium | Medium |
| Momentum | High | High | Medium |
| AdaGrad | Medium | High | Low |
| RMSProp | High | High | High |
| Adam | Very High | Very High | Very High |
| AdamW | Very High | Very High | State-of-the-Art |

---

# Which Optimizer Should You Use?

### ANN

```text
Adam
```

---

### CNN

```text
Adam
or
SGD + Momentum
```

---

### Transformers

```text
AdamW
```

---

### Research Experiments

```text
Adam
```

is usually the starting choice.

---

# TensorFlow Example

## Adam

```python
model.compile(
    optimizer='adam',
    loss='binary_crossentropy'
)
```

---

## SGD

```python
from tensorflow.keras.optimizers import SGD

optimizer = SGD(
    learning_rate=0.01,
    momentum=0.9
)
```

---

# PyTorch Example

```python
import torch.optim as optim

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)
```

---

# Industry Perspective

Modern AI systems often use:

### CNNs

```text
SGD + Momentum
```

### Transformers

```text
AdamW
```

### General Deep Learning

```text
Adam
```

---

# Real World Applications

Optimizers are used in:

- Computer Vision
- NLP
- Speech Recognition
- Healthcare AI
- Recommendation Systems
- Autonomous Vehicles

Every neural network requires an optimizer.

---

# Common Mistakes

❌ Learning Rate Too High

❌ Learning Rate Too Low

❌ Using SGD Without Momentum

❌ Ignoring Weight Decay

❌ Choosing Optimizer Without Experimentation

---

# Important Terms

| Term | Meaning |
|---------|---------|
| Optimizer | Algorithm that updates weights |
| Momentum | Uses previous gradients |
| RMSProp | Adaptive learning rate optimizer |
| Adam | Momentum + RMSProp |
| AdamW | Improved Adam with weight decay |
| Learning Rate | Step size during optimization |

---

# Interview Questions

## Q1. What is an optimizer?

An algorithm used to update model parameters and minimize the loss function.

---

## Q2. Why is Adam popular?

Because it combines Momentum and RMSProp, providing fast and stable convergence.

---

## Q3. Difference between Adam and AdamW?

AdamW handles weight decay separately, improving generalization.

---

## Q4. Which optimizer is commonly used in Transformers?

AdamW.

---

## Q5. What is the purpose of Momentum?

To accelerate convergence and reduce oscillations.

---

# Key Points Learned Today

✅ Learned Optimizers in Deep Learning

✅ Understood SGD

✅ Learned Momentum

✅ Understood AdaGrad

✅ Learned RMSProp

✅ Understood Adam

✅ Learned AdamW

✅ Studied Industry Best Practices

---

# Mini Summary

Optimizers are algorithms that update neural network parameters to minimize loss. While basic Gradient Descent provides the foundation, advanced optimizers such as Momentum, RMSProp, Adam, and AdamW significantly improve convergence speed, stability, and performance. Adam remains the most popular optimizer, while AdamW is the preferred choice for modern Transformer architectures.

---

# Progress

✅ Completed Day 28 - Optimizers in Deep Learning
