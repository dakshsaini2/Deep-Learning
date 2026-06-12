# Day 29 - Exponentially Weighted Moving Average (EWMA) in Deep Learning

## Introduction

When training neural networks, gradients often fluctuate significantly.

These fluctuations can cause:

- Slow convergence
- Noisy optimization
- Oscillations during training

To reduce these fluctuations, Deep Learning uses a technique called:

```text
Exponentially Weighted Moving Average (EWMA)
```

EWMA is a fundamental concept behind modern optimizers such as:

- Momentum
- RMSProp
- Adam
- AdamW

Understanding EWMA is essential because it forms the mathematical foundation of most advanced optimization algorithms.

---

# What is an Exponentially Weighted Moving Average?

EWMA is a method of calculating an average that:

```text
Gives More Importance
to Recent Values
```

and less importance to older values.

Unlike a simple average:

```text
All values are NOT treated equally.
```

---

# Why Do We Need EWMA?

Consider daily temperature data:

```text
20, 25, 30, 22, 28
```

A simple average gives equal importance to all observations.

However:

```text
Recent observations
are often more important.
```

EWMA captures this idea.

---

# Intuition

Imagine tracking stock prices.

Recent prices:

```text
More Relevant
```

than prices from months ago.

Similarly:

Recent gradients provide more useful information than very old gradients.

---

# EWMA Formula

The Exponentially Weighted Moving Average is computed as:



Where:

- Vₜ = Current EWMA
- Vₜ₋₁ = Previous EWMA
- Xₜ = Current Value
- β = Smoothing Parameter

---

# Understanding Beta (β)

Beta controls:

```text
How Much Past Information
is Remembered
```

Typical values:

| β Value | Behavior |
|----------|----------|
| 0.5 | Less Smoothing |
| 0.9 | Moderate Smoothing |
| 0.99 | Strong Smoothing |
| 0.999 | Very Strong Smoothing |

---

# Example Calculation

Suppose:

```text
β = 0.9

V₀ = 0

X₁ = 10
```

Calculation:

```text
V₁ = 0.9(0) + 0.1(10)

V₁ = 1
```

Next value:

```text
X₂ = 20
```

```text
V₂ = 0.9(1) + 0.1(20)

V₂ = 2.9
```

Notice:

```text
Recent values
influence the average more.
```

---

# Why is it Called Exponential?

Expanding the formula:

```text
Vt =
(1−β)Xt
+
β(1−β)Xt−1
+
β²(1−β)Xt−2
+
β³(1−β)Xt−3
...
```

Older observations receive exponentially smaller weights.

---

# Weight Distribution Example

For:

```text
β = 0.9
```

Weights become:

| Observation | Weight |
|-------------|---------|
| Current | 0.10 |
| Previous | 0.09 |
| Older | 0.081 |
| Older | 0.0729 |

Weights decrease exponentially.

---

# Effect of Different Beta Values

### Small Beta

```text
β = 0.5
```

Result:

```text
Fast Response
Less Smoothing
```

---

### Large Beta

```text
β = 0.99
```

Result:

```text
Slow Response
More Smoothing
```

---

# Visualization

Small Beta:

```text
Noisy Curve
```

Large Beta:

```text
Smooth Curve
```

---

# Relationship with Neural Networks

During training:

Gradients often look like:

```text
3
-5
7
-4
8
```

Such fluctuations create instability.

EWMA smooths these gradients.

---

# EWMA in Optimization

Instead of using:

```text
Current Gradient Only
```

we use:

```text
Historical Gradient Information
```

This produces:

- Smoother Updates
- Faster Convergence

---

# Momentum Optimizer

Momentum uses EWMA of gradients.

Formula:



Where:

- gₜ = Current Gradient

Weight Update:



---

# Why Momentum Works?

Without Momentum:

```text
Gradient
changes direction frequently
```

With Momentum:

```text
Optimization becomes smoother
```

and moves consistently toward the minimum.

---

# RMSProp and EWMA

RMSProp uses EWMA of:

```text
Squared Gradients
```

Formula:



This helps adapt learning rates.

---

# Adam and EWMA

Adam combines:

1. EWMA of Gradients
2. EWMA of Squared Gradients

First Moment:



Second Moment:



---

# Bias Correction Problem

At the beginning:

```text
V₀ = 0
```

Therefore EWMA is biased toward zero.

This is called:

```text
Initialization Bias
```

---

# Bias Correction Formula

Corrected Estimate:



This provides a more accurate estimate.

---

# Why Bias Correction Matters?

Especially during:

```text
Early Training Stages
```

when moving averages are still developing.

Adam uses bias correction extensively.

---

# Advantages of EWMA

✅ Smooths Noisy Data

✅ Reduces Oscillations

✅ Faster Convergence

✅ Foundation of Modern Optimizers

✅ Easy to Compute

---

# Limitations

❌ Requires Choosing β

❌ Large β Responds Slowly

❌ Small β Produces Noisy Estimates

---

# Industry Perspective

Almost every modern optimizer uses EWMA:

- Momentum
- RMSProp
- Adam
- AdamW

Without EWMA:

```text
Modern Deep Learning
would train much slower.
```

---

# TensorFlow Example

```python
optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.999
)
```

Here:

```text
beta_1
and
beta_2
```

are EWMA coefficients.

---

# Real World Applications

EWMA is used in:

- Deep Learning Optimizers
- Financial Forecasting
- Signal Processing
- Time Series Analysis
- Control Systems

---

# Important Terms

| Term | Meaning |
|---------|---------|
| EWMA | Exponentially Weighted Moving Average |
| β | Smoothing Factor |
| Momentum | EWMA of Gradients |
| RMSProp | EWMA of Squared Gradients |
| Adam | Combination of Two EWMAs |
| Bias Correction | Removes Initialization Bias |

---

# Interview Questions

## Q1. What is EWMA?

A moving average that gives higher weight to recent observations.

---

## Q2. Why is EWMA important in Deep Learning?

It smooths gradients and improves optimization.

---

## Q3. Which optimizers use EWMA?

Momentum, RMSProp, Adam, and AdamW.

---

## Q4. What does β control?

The amount of smoothing and memory of past values.

---

## Q5. Why is bias correction needed?

Because moving averages are initially biased toward zero.

---

# Key Points Learned Today

✅ Learned EWMA

✅ Understood Smoothing

✅ Learned Role of Beta

✅ Understood Momentum

✅ Learned RMSProp Foundation

✅ Understood Adam's Mathematics

✅ Learned Bias Correction

---

# Mini Summary

Exponentially Weighted Moving Average (EWMA) is a technique that assigns greater importance to recent observations while gradually reducing the influence of older values. It is the mathematical foundation of Momentum, RMSProp, Adam, and AdamW optimizers, making it one of the most important concepts in modern Deep Learning optimization.

---
#d
# Progress

✅ Completed Day 29 - Exponentially Weighted Moving Average (EWMA)
