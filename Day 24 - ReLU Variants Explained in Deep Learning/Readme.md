# Day 24 - ReLU Variants Explained in Deep Learning

## Introduction

The introduction of ReLU (Rectified Linear Unit) revolutionized Deep Learning.

Before ReLU, neural networks commonly used:

- Sigmoid
- Tanh

These activation functions suffered from:

```text
Vanishing Gradient Problem
```

which made training deep neural networks difficult.

ReLU solved many of these issues and became the default activation function in modern Deep Learning.

However, ReLU itself has limitations.

To overcome these limitations, researchers developed several ReLU variants.

This chapter covers:

- ReLU
- Dying ReLU Problem
- Leaky ReLU
- Parametric ReLU (PReLU)
- Randomized ReLU (RReLU)
- ELU
- SELU
- GELU
- Comparison of ReLU Variants

---

# What is ReLU?

ReLU stands for:

```text
Rectified Linear Unit
```

Formula:



---

# ReLU Behavior

```text
x < 0 → 0

x > 0 → x
```

Example:

| Input | Output |
|---------|---------|
| -5 | 0 |
| -2 | 0 |
| 0 | 0 |
| 3 | 3 |
| 7 | 7 |

---

# Why ReLU Became Popular?

Advantages:

✅ Computationally Fast

✅ Easy to Implement

✅ Reduces Vanishing Gradients

✅ Enables Deep Networks

✅ Sparse Activations

---

# ReLU Visualization

```text
        /
       /
      /
-----/
```

---

# Problem with ReLU

ReLU suffers from:

```text
Dying ReLU Problem
```

---

# Dying ReLU Problem

If a neuron always receives:

```text
Negative Inputs
```

its output becomes:

```text
0
```

and gradient becomes:

```text
0
```

The neuron stops learning forever.

Such neurons are called:

```text
Dead Neurons
```

---

# Why Dying ReLU Happens?

Large negative weight updates may push neurons into:

```text
Always Negative Region
```

Once this happens:

```text
Output = 0
Gradient = 0
```

Learning stops.

---

# Solution: ReLU Variants

Researchers developed improved versions of ReLU.

Major variants:

1. Leaky ReLU
2. PReLU
3. RReLU
4. ELU
5. SELU
6. GELU

---

# 1. Leaky ReLU

Leaky ReLU allows a small gradient for negative inputs.

Formula:



---

# Leaky ReLU Behavior

```text
Positive Input → Normal ReLU

Negative Input → Small Slope
```

Example:

| Input | Output |
|---------|---------|
| -10 | -0.1 |
| -5 | -0.05 |
| 5 | 5 |

---

# Advantages of Leaky ReLU

✅ Solves Dying ReLU

✅ Better Gradient Flow

✅ Easy Replacement for ReLU

---

# Limitation

Slope value:

```text
0.01
```

is manually chosen.

Not always optimal.

---

# 2. Parametric ReLU (PReLU)

PReLU learns the negative slope automatically.

Formula:



Where:

```text
a = Learnable Parameter
```

---

# Why PReLU?

Instead of fixing:

```text
0.01
```

the network learns the best slope.

Benefits:

✅ More Flexible

✅ Better Performance

✅ Adaptive Learning

---

# Limitation

Introduces additional trainable parameters.

Slightly increases model complexity.

---

# 3. Randomized ReLU (RReLU)

RReLU randomly chooses negative slopes during training.

Formula:

```text
a ~ Random Distribution
```

---

# Why RReLU?

Acts as:

```text
Regularization
```

and helps reduce overfitting.

Benefits:

✅ Improved Generalization

✅ Better Robustness

---

# Limitation

Less commonly used in production systems.

---

# 4. ELU (Exponential Linear Unit)

ELU introduces smooth negative outputs.

Formula:

```text
x                 if x > 0

α(e^x - 1)        if x ≤ 0
```

---

# ELU Behavior

Positive Region:

```text
Same as ReLU
```

Negative Region:

```text
Smooth Exponential Curve
```

---

# Advantages of ELU

✅ Reduces Bias Shift

✅ Faster Learning

✅ Better Gradient Flow

✅ Smooth Negative Outputs

---

# Why ELU is Better Than ReLU?

ReLU completely ignores negative values.

ELU still learns useful information from:

```text
Negative Activations
```

---

# 5. SELU (Scaled Exponential Linear Unit)

Improved version of ELU.

Formula:

```text
SELU = λ × ELU(x)
```

where:

```text
λ = Scaling Constant
```

---

# Key Idea Behind SELU

SELU creates:

```text
Self-Normalizing Neural Networks
```

Activations automatically maintain:

```text
Mean ≈ 0
Variance ≈ 1
```

---

# Benefits of SELU

✅ Stable Training

✅ Faster Convergence

✅ Less Need for BatchNorm

---

# Limitation

Requires:

- Specific Initialization
- Specific Architecture Conditions

---

# 6. GELU (Gaussian Error Linear Unit)

Modern activation used in:

- BERT
- GPT
- Transformers

Approximation:

```text
GELU(x)
≈
0.5x(1+tanh(...))
```

---

# Intuition Behind GELU

Instead of:

```text
Hard Threshold
```

like ReLU,

GELU uses:

```text
Probabilistic Gating
```

---

# Why GELU Became Popular?

Advantages:

✅ Smooth Activation

✅ Better Gradient Flow

✅ Superior Transformer Performance

✅ State-of-the-Art Results

---

# ReLU vs GELU

| ReLU | GELU |
|--------|--------|
| Hard Decision | Smooth Decision |
| Faster | Slightly Slower |
| CNNs | Transformers |
| Simpler | More Powerful |

---

# Activation Function Evolution

```text
Sigmoid
   ↓
Tanh
   ↓
ReLU
   ↓
Leaky ReLU
   ↓
PReLU
   ↓
ELU
   ↓
SELU
   ↓
GELU
```

---

# Comparison Table

| Activation | Solves Dying ReLU | Learnable | Modern Usage |
|------------|-------------------|------------|--------------|
| ReLU | ❌ | ❌ | Very High |
| Leaky ReLU | ✅ | ❌ | High |
| PReLU | ✅ | ✅ | Moderate |
| RReLU | ✅ | Random | Low |
| ELU | ✅ | ❌ | Moderate |
| SELU | ✅ | ❌ | Moderate |
| GELU | ✅ | ❌ | Very High |

---

# Which Activation Should You Use?

### ANN

```text
ReLU
Leaky ReLU
```

---

### CNN

```text
ReLU
Leaky ReLU
```

---

### Deep Networks

```text
ELU
SELU
```

---

### Transformers

```text
GELU
```

---

# TensorFlow Examples

## ReLU

```python
Dense(128, activation='relu')
```

---

## Leaky ReLU

```python
from tensorflow.keras.layers import LeakyReLU

LeakyReLU(alpha=0.01)
```

---

## PReLU

```python
from tensorflow.keras.layers import PReLU

PReLU()
```

---

## ELU

```python
Dense(128, activation='elu')
```

---

# Industry Perspective

Most modern production systems use:

### Computer Vision

```text
ReLU
Leaky ReLU
```

### NLP & LLMs

```text
GELU
```

Models using GELU:

- :contentReference[oaicite:3]{index=3}
- :contentReference[oaicite:4]{index=4}

---

# Interview Questions

## Q1. What problem does Leaky ReLU solve?

The Dying ReLU Problem.

---

## Q2. Difference between ReLU and PReLU?

PReLU learns the negative slope automatically.

---

## Q3. Why is GELU used in Transformers?

Because it provides smoother activations and better gradient flow.

---

## Q4. What is SELU known for?

Self-Normalizing Neural Networks.

---

## Q5. Which activation is most popular today?

ReLU for CNNs and GELU for Transformers.

---

# Key Points Learned Today

✅ Learned ReLU and Dying ReLU

✅ Understood Leaky ReLU

✅ Learned PReLU and RReLU

✅ Understood ELU and SELU

✅ Learned GELU

✅ Compared all major ReLU variants

✅ Studied modern industry usage

---

# Mini Summary

ReLU transformed Deep Learning by solving many optimization problems, but it introduced the Dying ReLU issue. Advanced variants such as Leaky ReLU, PReLU, ELU, SELU, and GELU were developed to improve gradient flow, training stability, and model performance. Today, ReLU dominates Computer Vision while GELU powers modern Transformer-based models like GPT and BERT.

---

# Progress

✅ Completed Day 24 - ReLU Variants Explained
