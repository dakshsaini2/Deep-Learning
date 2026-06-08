# Day 25 - Weight Initialization Techniques in Deep Learning

## Introduction

Weight Initialization is one of the most critical yet often overlooked concepts in Deep Learning.

Before a neural network starts learning, its weights must be initialized.

The choice of initialization affects:

- Training Speed
- Convergence
- Gradient Flow
- Model Accuracy
- Stability of Learning

Poor initialization can cause:

```text
Vanishing Gradients
Exploding Gradients
Slow Convergence
Training Failure
```

Modern Deep Learning owes much of its success to better weight initialization techniques.

This chapter covers:

- Why Weight Initialization Matters
- Problems with Poor Initialization
- Zero Initialization
- Random Initialization
- Xavier Initialization
- He Initialization
- Modern Best Practices

---

# What is Weight Initialization?

Before training begins:

```text
Weights = ?
```

The network needs initial values for all weights.

Weight Initialization is the process of assigning these initial values.

Example:

```text
Input Layer
      ↓
Hidden Layer
      ↓
Output Layer
```

Each connection requires a starting weight.

---

# Why is Weight Initialization Important?

Neural Networks learn using:

```text
Forward Propagation
↓
Loss Calculation
↓
Backpropagation
↓
Weight Updates
```

Poor initialization disrupts this process.

Good initialization helps:

✅ Faster Learning

✅ Stable Gradients

✅ Better Convergence

✅ Improved Performance

---

# Goal of Weight Initialization

The ideal initialization should:

```text
Maintain Information Flow
```

throughout the network.

Meaning:

```text
Activations should not become:
Too Small
or
Too Large
```

---

# Problem 1: Zero Initialization

Suppose:

```text
All Weights = 0
```

Example:

```text
W1 = 0
W2 = 0
W3 = 0
```

---

# Why Zero Initialization Fails?

All neurons become identical.

During training:

```text
Same Input
↓
Same Output
↓
Same Gradient
↓
Same Update
```

All neurons learn exactly the same thing.

This is called:

```text
Symmetry Problem
```

---

# Symmetry Problem

Instead of learning:

```text
Different Features
```

all neurons learn:

```text
Identical Features
```

Result:

```text
Network Capacity Wasted
```

---

# Problem 2: Very Large Weights

Suppose:

```text
Weights = 100
```

Forward propagation generates:

```text
Huge Activations
```

This can cause:

```text
Exploding Activations
```

---

# Consequences

Large activations produce:

```text
Exploding Gradients
```

Training becomes unstable.

Loss may become:

```text
NaN
```

---

# Problem 3: Very Small Weights

Suppose:

```text
Weights = 0.000001
```

Outputs become extremely small.

This causes:

```text
Vanishing Gradients
```

Learning becomes very slow.

---

# Random Initialization

Simple solution:

```text
Initialize Random Weights
```

Example:

```text
W = [-0.2, 0.1, 0.05]
```

---

# Advantages

✅ Breaks Symmetry

✅ Different Neurons Learn Different Features

---

# Limitation

Random initialization alone does not solve:

- Vanishing Gradients
- Exploding Gradients

---

# Why Gradient Problems Occur?

Deep networks repeatedly perform:

```text
Matrix Multiplication
```

through many layers.

If activations grow:

```text
Exploding Gradient
```

If activations shrink:

```text
Vanishing Gradient
```

---

# Desired Property

We want:

```text
Variance of Activations
≈ Constant
```

across all layers.

This led to advanced initialization methods.

---

# Xavier Initialization

Introduced by:

```text
Xavier Glorot
```

One of the biggest breakthroughs in Deep Learning.

---

# Main Idea

Initialize weights so that:

```text
Activation Variance
remains stable
```

across layers.

---

# Xavier Formula

Weights are sampled from:

```text
Variance = 1 / n
```

Where:

```text
n = Number of Input Connections
```

---

# Xavier Initialization Intuition

Goal:

```text
Neither Too Large
Nor Too Small
```

activations.

Benefits:

✅ Stable Forward Pass

✅ Stable Backpropagation

✅ Faster Convergence

---

# When to Use Xavier?

Best suited for:

- Sigmoid
- Tanh

activation functions.

---

# Limitation

ReLU activates only half of neurons.

Therefore Xavier is not optimal for ReLU.

---

# He Initialization

Introduced by:

```text
Kaiming He
```

Specifically designed for:

```text
ReLU Networks
```

---

# Main Idea

Compensate for neurons that become inactive under ReLU.

---

# He Formula

Weights are initialized using:

```text
Variance = 2 / n
```

Where:

```text
n = Number of Input Connections
```

---

# Why He Works Better for ReLU?

ReLU removes all negative activations.

Approximately:

```text
50% Information Lost
```

He Initialization compensates for this loss.

---

# Benefits of He Initialization

✅ Better Gradient Flow

✅ Faster Training

✅ Stable Deep Networks

✅ Reduced Vanishing Gradients

---

# Xavier vs He Initialization

| Feature | Xavier | He |
|----------|--------|----|
| Designed For | Sigmoid/Tanh | ReLU |
| Variance | 1/n | 2/n |
| Deep Networks | Good | Better |
| Modern Usage | Moderate | Very High |

---

# Visual Intuition

Bad Initialization:

```text
Layer 1 → Huge Values
Layer 2 → Huge Values
Layer 3 → Explosion
```

OR

```text
Layer 1 → Tiny Values
Layer 2 → Smaller Values
Layer 3 → Vanish
```

Good Initialization:

```text
Layer 1 → Stable
Layer 2 → Stable
Layer 3 → Stable
```

---

# Weight Initialization and Activation Functions

Recommended combinations:

| Activation | Initialization |
|------------|---------------|
| Sigmoid | Xavier |
| Tanh | Xavier |
| ReLU | He |
| Leaky ReLU | He |
| GELU | He |

---

# Weight Initialization in Modern Deep Learning

Modern architectures use:

- He Initialization
- Xavier Initialization
- Layer Normalization
- Batch Normalization

together.

---

# TensorFlow Example

## Xavier Initialization

```python
from tensorflow.keras.initializers import GlorotUniform

Dense(
    128,
    activation='tanh',
    kernel_initializer=GlorotUniform()
)
```

---

## He Initialization

```python
from tensorflow.keras.initializers import HeNormal

Dense(
    128,
    activation='relu',
    kernel_initializer=HeNormal()
)
```

---

# PyTorch Example

```python
import torch.nn.init as init

init.xavier_uniform_(layer.weight)

init.kaiming_normal_(layer.weight)
```

---

# Industry Perspective

Almost all modern Deep Learning systems use:

```text
ReLU + He Initialization
```

or

```text
GELU + He Initialization
```

Examples include:

- CNNs
- ResNets
- Vision Transformers
- Large Language Models

---

# Common Mistakes

❌ Initializing all weights to zero

❌ Using huge random values

❌ Using tiny random values

❌ Using Xavier with deep ReLU networks

❌ Ignoring activation function choice

---

# Important Terms

| Term | Meaning |
|--------|---------|
| Weight Initialization | Assigning initial weights |
| Symmetry Problem | Identical neuron learning |
| Xavier Initialization | Initialization for Sigmoid/Tanh |
| He Initialization | Initialization for ReLU |
| Vanishing Gradient | Tiny gradients |
| Exploding Gradient | Huge gradients |

---

# Interview Questions

## Q1. Why can't we initialize all weights to zero?

Because all neurons learn identical features due to the symmetry problem.

---

## Q2. What is Xavier Initialization?

A weight initialization method designed to maintain activation variance for Sigmoid and Tanh networks.

---

## Q3. Why is He Initialization better for ReLU?

Because it compensates for the loss of negative activations.

---

## Q4. Which initialization is used most often today?

He Initialization.

---

## Q5. What problems can poor initialization cause?

Vanishing gradients, exploding gradients, and slow convergence.

---

# Key Points Learned Today

✅ Learned Weight Initialization

✅ Understood Symmetry Problem

✅ Learned Random Initialization

✅ Understood Xavier Initialization

✅ Learned He Initialization

✅ Studied Activation-Initialization Pairing

✅ Learned Modern Industry Practices

---

# Mini Summary

Weight Initialization determines how a neural network begins learning. Poor initialization can lead to vanishing gradients, exploding gradients, and slow convergence. Modern Deep Learning primarily relies on Xavier Initialization for Sigmoid/Tanh networks and He Initialization for ReLU-based architectures to ensure stable and efficient training.

---

# Progress

✅ Completed Day 25 - Weight Initialization Techniques
