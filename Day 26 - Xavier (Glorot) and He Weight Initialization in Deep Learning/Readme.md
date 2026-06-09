# Day 26 - Xavier (Glorot) and He Weight Initialization in Deep Learning

## Introduction

One of the most important factors affecting neural network training is:

```text
Weight Initialization
```

Before training begins, every neuron must be assigned initial weights.

Poor initialization can cause:

- Vanishing Gradients
- Exploding Gradients
- Slow Convergence
- Poor Performance

To solve these issues, researchers developed advanced initialization techniques:

1. Xavier Initialization (Glorot Initialization)
2. He Initialization (Kaiming Initialization)

These methods have become the industry standard for modern Deep Learning.

---

# Why Weight Initialization Matters?

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

If weights are poorly initialized:

```text
Learning becomes unstable
```

or may completely fail.

---

# Problems with Bad Initialization

## Very Small Weights

Example:

```text
W = 0.000001
```

Result:

```text
Tiny Activations
↓
Tiny Gradients
↓
Vanishing Gradient Problem
```

---

## Very Large Weights

Example:

```text
W = 100
```

Result:

```text
Huge Activations
↓
Huge Gradients
↓
Exploding Gradient Problem
```

---

# Goal of Good Initialization

The objective is:

```text
Stable Activations
Stable Gradients
```

throughout the network.

We want:

```text
Signal Strength
```

to remain approximately constant across layers.

---

# Xavier Initialization

Also called:

```text
Glorot Initialization
```

because it was introduced by:

```text
Xavier Glorot
```

and

```text
Yoshua Bengio
```

in 2010.

---

# Important Note

```text
Xavier Initialization
=
Glorot Initialization
```

Both names refer to the same technique.

---

# Why Xavier Initialization Was Needed?

Traditional random initialization often caused:

- Vanishing Gradients
- Exploding Gradients

especially in deep networks.

Xavier Initialization was designed to maintain:

```text
Activation Variance
```

across layers.

---

# Xavier Initialization Formula

For a uniform distribution:



Where:

- `n_in` = Number of input neurons
- `n_out` = Number of output neurons

---

# Xavier Variance

For normal distribution:



---

# Intuition Behind Xavier Initialization

The goal is:

```text
Neither Too Large
Nor Too Small
```

activations.

This keeps:

```text
Forward Propagation Stable
Backpropagation Stable
```

---

# Best Activation Functions for Xavier

Recommended:

✅ Sigmoid

✅ Tanh

Because Xavier was specifically designed for these activations.

---

# Limitation of Xavier

ReLU behaves differently.

ReLU removes all negative values:



Approximately:

```text
50% Activations Become Zero
```

Therefore Xavier may not preserve variance effectively in ReLU networks.

---

# He Initialization

Introduced by:

```text
Kaiming He
```

and colleagues in 2015.

Also called:

```text
Kaiming Initialization
```

---

# Why He Initialization Was Developed?

Researchers observed:

```text
Xavier Works Well
```

for:

- Sigmoid
- Tanh

but not perfectly for:

```text
ReLU
```

He Initialization was specifically designed for ReLU-based networks.

---

# He Initialization Formula

For normal distribution:



---

# He Uniform Formula



---

# Why He Initialization Works?

ReLU removes approximately:

```text
50% of activations
```

He Initialization compensates for this information loss by using:

```text
Larger Initial Variance
```

than Xavier.

---

# Advantages of He Initialization

✅ Better Gradient Flow

✅ Faster Convergence

✅ Stable Deep Networks

✅ Reduced Vanishing Gradients

✅ Ideal for ReLU Family

---

# Xavier vs He Initialization

| Feature | Xavier (Glorot) | He (Kaiming) |
|----------|----------|----------|
| Introduced | 2010 | 2015 |
| Inventor | Xavier Glorot | Kaiming He |
| Variance | 2/(n_in+n_out) | 2/n_in |
| Best For | Sigmoid, Tanh | ReLU, Leaky ReLU |
| Deep Networks | Good | Better |
| Modern Usage | Moderate | Very High |

---

# Visualization

Bad Initialization:

```text
Layer 1 → Small Values
Layer 2 → Smaller Values
Layer 3 → Vanished
```

Result:

```text
Vanishing Gradients
```

---

Another possibility:

```text
Layer 1 → Large Values
Layer 2 → Larger Values
Layer 3 → Exploded
```

Result:

```text
Exploding Gradients
```

---

Good Initialization:

```text
Layer 1 → Stable
Layer 2 → Stable
Layer 3 → Stable
```

Result:

```text
Efficient Learning
```

---

# Activation Function and Initialization Pairing

| Activation Function | Recommended Initialization |
|--------------------|---------------------------|
| Sigmoid | Xavier |
| Tanh | Xavier |
| ReLU | He |
| Leaky ReLU | He |
| ELU | He |
| GELU | He |

---

# TensorFlow Example

## Xavier (Glorot)

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

Modern AI systems typically use:

```text
ReLU + He Initialization
```

or

```text
GELU + He Initialization
```

Examples:

- CNNs
- ResNets
- Vision Transformers
- Large Language Models

Xavier is still widely used in:

- Older architectures
- Sigmoid/Tanh networks

---

# Common Mistakes

❌ Initializing all weights to zero

❌ Using random values without considering activation functions

❌ Using Xavier with deep ReLU networks

❌ Ignoring gradient stability

---

# Important Terms

| Term | Meaning |
|---------|---------|
| Weight Initialization | Initial assignment of weights |
| Xavier Initialization | Initialization for Sigmoid/Tanh |
| Glorot Initialization | Another name for Xavier Initialization |
| He Initialization | Initialization for ReLU |
| Vanishing Gradient | Extremely small gradients |
| Exploding Gradient | Extremely large gradients |

---

# Interview Questions

## Q1. Are Xavier and Glorot Initialization different?

No. Xavier Initialization and Glorot Initialization are the same technique.

---

## Q2. Why was He Initialization introduced?

To improve initialization for ReLU-based networks.

---

## Q3. Which initialization is best for Sigmoid and Tanh?

Xavier (Glorot) Initialization.

---

## Q4. Which initialization is best for ReLU?

He (Kaiming) Initialization.

---

## Q5. Why is weight initialization important?

Because it affects gradient flow, convergence speed, and overall network performance.

---

# Key Points Learned Today

✅ Learned Xavier Initialization

✅ Understood Glorot Initialization

✅ Learned He Initialization

✅ Understood activation-initialization pairing

✅ Learned variance preservation

✅ Studied modern industry practices

✅ Understood gradient stability

---

# Mini Summary

Weight initialization plays a crucial role in neural network training. Xavier (Glorot) Initialization was designed for Sigmoid and Tanh activations, while He Initialization was developed specifically for ReLU-based networks. Both techniques help maintain stable activations and gradients, leading to faster convergence and improved model performance.

---

# Progress

✅ Completed Day 26 - Xavier (Glorot) and He Weight Initialization
