# Day 30 - SGD with Momentum in Deep Learning

## Introduction

In Day 28, we learned about optimizers and discovered that:

```text
Stochastic Gradient Descent (SGD)
```

is one of the most fundamental optimization algorithms in Deep Learning.

However, SGD has a major drawback:

```text
Noisy Updates
```

which can make training slow and unstable.

To overcome this problem, researchers introduced:

```text
Momentum
```

The combination of SGD and Momentum became one of the most important optimization techniques in Deep Learning.

This chapter covers:

- SGD Recap
- Why SGD Struggles
- Momentum Concept
- SGD with Momentum
- Mathematical Formulation
- Advantages and Applications

---

# Recap: What is SGD?

Stochastic Gradient Descent updates weights using:

```text
One Training Sample
```

at a time.

Update Rule:



Where:

- W = Weight
- η = Learning Rate
- gₜ = Current Gradient

---

# Problem with SGD

SGD updates depend only on:

```text
Current Gradient
```

As a result:

```text
Optimization Path
becomes noisy
```

Example:

```text
Left
Right
Left
Right
Forward
Backward
```

instead of moving directly toward the optimum.

---

# Visualization of SGD

```text
Goal
  *
 /\
/  \

SGD Path:

\/\/\/\/\/\/
```

Many unnecessary oscillations occur.

---

# Why is This a Problem?

Consequences:

❌ Slow Convergence

❌ Wasted Computation

❌ Difficulty Escaping Flat Regions

❌ Unstable Learning

---

# Inspiration Behind Momentum

Momentum comes from Physics.

Imagine pushing a heavy ball downhill.

Initially:

```text
Slow Movement
```

After gaining speed:

```text
Momentum Builds Up
```

and the ball continues moving smoothly.

Neural network optimization uses the same idea.

---

# What is Momentum?

Momentum allows the optimizer to:

```text
Remember Previous Gradients
```

instead of relying only on the current gradient.

This creates smoother updates.

---

# Core Idea

Instead of:

```text
Current Gradient Only
```

use:

```text
Current Gradient
+
Past Gradient Information
```

---

# Momentum Formula

Velocity Term:



Where:

- vₜ = Velocity
- β = Momentum Coefficient
- gₜ = Current Gradient

---

# Weight Update Rule

Weights update using:



Instead of directly using the current gradient.

---

# Understanding Beta (β)

β controls:

```text
How Much History
is Remembered
```

Common values:

| β | Behavior |
|----|----------|
| 0.5 | Short Memory |
| 0.9 | Standard |
| 0.99 | Long Memory |
| 0.999 | Very Long Memory |

Most commonly used:

```text
β = 0.9
```

---

# Why Momentum Works?

Without Momentum:

```text
Each Step
Forgets Previous Steps
```

With Momentum:

```text
Past Directions
Influence Future Directions
```

This creates smoother movement.

---

# SGD vs SGD with Momentum

## Standard SGD

```text
Current Gradient
↓
Update
```

---

## SGD + Momentum

```text
Past Gradients
      ↓
Current Gradient
      ↓
Velocity
      ↓
Update
```

---

# Mountain Analogy

Imagine descending a mountain.

### SGD

```text
Walk
Stop
Walk
Stop
```

Movement is inefficient.

---

### SGD with Momentum

```text
Roll a Ball
```

The ball naturally gains speed and moves more efficiently.

---

# Effect on Optimization

Without Momentum:

```text
Zig-Zag Motion
```

With Momentum:

```text
Smooth Motion
```

toward the minimum.

---

# Relationship with EWMA

Momentum uses:

```text
Exponentially Weighted Moving Average (EWMA)
```

of gradients.

Formula:



This is exactly the EWMA formula learned in Day 29.

---

# Advantages of SGD with Momentum

## 1. Faster Convergence

Optimization reaches the minimum faster.

---

## 2. Reduced Oscillation

Smooths noisy gradient updates.

---

## 3. Better Gradient Flow

Works better in deep networks.

---

## 4. Escapes Shallow Local Minima

Momentum helps move through flat regions.

---

## 5. More Stable Training

Produces smoother learning curves.

---

# Practical Example

Suppose:

```text
Learning Rate = 0.01
Momentum = 0.9
```

Current Gradient:

```text
g = 5
```

Velocity accumulates information from previous gradients and produces a more informed update.

This usually performs much better than plain SGD.

---

# Nesterov Momentum

An improved version of Momentum.

Idea:

```text
Look Ahead
Before Updating
```

Instead of reacting only to the current position.

---

# Why Nesterov Works Better?

Standard Momentum:

```text
Move
Then Correct
```

Nesterov:

```text
Look Ahead
Then Move
```

Often converges faster.

---

# SGD with Momentum vs Adam

| SGD + Momentum | Adam |
|---------------|------|
| Simpler | More Complex |
| Less Memory | More Memory |
| Excellent for CNNs | Excellent for General Tasks |
| Needs Learning Rate Tuning | More Adaptive |

---

# Industry Usage

### Computer Vision

Common choice:

```text
SGD + Momentum
```

Used in:

- ResNet
- EfficientNet
- Object Detection Models

---

### NLP and Transformers

Often use:

```text
Adam
or
AdamW
```

instead.

---

# TensorFlow Example

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

optimizer = optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9
)
```

---

# Hyperparameters

Common settings:

| Parameter | Value |
|------------|--------|
| Learning Rate | 0.01 |
| Momentum | 0.9 |
| Weight Decay | 0.0001 |

---

# Real World Applications

SGD with Momentum is widely used in:

- Image Classification
- Object Detection
- Face Recognition
- Medical Imaging
- Autonomous Vehicles

---

# Important Terms

| Term | Meaning |
|---------|---------|
| SGD | Stochastic Gradient Descent |
| Momentum | Uses previous gradients |
| Velocity | Smoothed gradient estimate |
| β | Momentum coefficient |
| EWMA | Exponentially Weighted Moving Average |
| Nesterov | Improved Momentum method |

---

# Interview Questions

## Q1. What problem does Momentum solve?

It reduces oscillations and accelerates convergence.

---

## Q2. What is the role of β in Momentum?

It controls how much past gradient information is remembered.

---

## Q3. What is the most common value of Momentum?

```text
0.9
```

---

## Q4. How is Momentum related to EWMA?

Momentum uses an exponentially weighted moving average of gradients.

---

## Q5. Why is SGD with Momentum still popular?

Because it is simple, efficient, and performs extremely well in computer vision tasks.

---

# Key Points Learned Today

✅ Learned SGD with Momentum

✅ Understood Optimization Problems

✅ Learned Velocity Concept

✅ Understood EWMA Connection

✅ Learned Nesterov Momentum

✅ Compared SGD with Adam

✅ Studied Industry Applications

---

# Mini Summary

SGD with Momentum improves standard Stochastic Gradient Descent by incorporating past gradient information using an Exponentially Weighted Moving Average (EWMA). This reduces oscillations, accelerates convergence, and leads to more stable optimization. It remains one of the most widely used optimizers in Computer Vision and Deep Learning.

---

# Progress

✅ Completed Day 30 - SGD with Momentum in Deep Learning
