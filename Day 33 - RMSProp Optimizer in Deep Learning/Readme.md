# Day 33 - RMSProp Optimizer in Deep Learning

## Introduction

In Day 32, we learned about:

```text
AdaGrad (Adaptive Gradient Algorithm)
```

AdaGrad introduced:

```text
Adaptive Learning Rates
```

for each parameter.

Although AdaGrad was a breakthrough, it had a major problem:

```text
Learning Rate Keeps Decreasing Forever
```

As training continues:

```text
Learning Rate → 0
```

Eventually:

```text
Learning Stops
```

To solve this issue, Geoffrey Hinton proposed:

```text
RMSProp
```

which stands for:

```text
Root Mean Square Propagation
```

RMSProp became one of the most influential optimizers and later inspired Adam.

---

# Why AdaGrad Fails?

AdaGrad accumulates all past gradients:



As training progresses:

```text
Gt becomes huge
```

Therefore:

```text
Learning Rate becomes tiny
```

Eventually:

```text
Weight Updates ≈ 0
```

Training slows dramatically.

---

# Core Idea of RMSProp

Instead of storing:

```text
All Past Gradients
```

RMSProp stores:

```text
Recent Gradients Only
```

using:

```text
Exponentially Weighted Moving Average (EWMA)
```

This prevents learning rates from shrinking forever.

---

# Intuition

Think of AdaGrad as:

```text
Remember Everything Forever
```

RMSProp says:

```text
Recent Information
is More Important
```

Old gradients gradually lose importance.

---

# RMSProp Formula

RMSProp maintains:



Where:

- Sₜ = Running Average of Squared Gradients
- β = Decay Rate
- gₜ = Current Gradient

---

# Weight Update Rule

Weights are updated using:



Where:

- η = Learning Rate
- ε = Small Constant

---

# Understanding the Formula

Suppose:

```text
Large Gradient
```

Then:

```text
St becomes large
```

Therefore:

```text
Update becomes smaller
```

---

# Conversely

If:

```text
Gradient is small
```

Then:

```text
St becomes smaller
```

and:

```text
Update becomes larger
```

This creates:

```text
Adaptive Learning Rates
```

---

# Role of Beta (β)

Beta controls:

```text
How Much History
is Remembered
```

Typical values:

| β | Meaning |
|----|---------|
| 0.9 | Standard |
| 0.95 | Longer Memory |
| 0.99 | Very Long Memory |

Most commonly used:

```text
β = 0.9
```

---

# RMSProp and EWMA

RMSProp directly uses:

```text
Exponentially Weighted Moving Average
```

Formula:



This allows:

```text
Recent Gradients
to influence learning more
```

than older gradients.

---

# Why RMSProp Works Better Than AdaGrad?

### AdaGrad

```text
Past Gradients
Never Forgotten
```

Result:

```text
Learning Rate Shrinks Forever
```

---

### RMSProp

```text
Older Gradients Fade Away
```

Result:

```text
Stable Learning Rate
```

throughout training.

---

# Visualization

### AdaGrad

```text
Learning Rate
↓
↓
↓
↓
Almost Zero
```

---

### RMSProp

```text
Learning Rate
≈ Stable
```

---

# Advantages of RMSProp

## 1. Adaptive Learning Rate

Each parameter gets its own learning rate.

---

## 2. Solves AdaGrad's Problem

No continuous decay to zero.

---

## 3. Faster Convergence

Optimization becomes more efficient.

---

## 4. Works Well for Deep Networks

Particularly effective for:

- CNNs
- RNNs
- Deep ANNs

---

## 5. Handles Non-Stationary Data

Useful when data distributions change.

---

# Example

Suppose:

```text
Learning Rate = 0.001
β = 0.9
```

Current Gradient:

```text
g = 5
```

RMSProp computes:

```text
Running Average
```

of recent squared gradients and adjusts learning automatically.

---

# RMSProp vs AdaGrad

| AdaGrad | RMSProp |
|----------|----------|
| Stores All Past Gradients | Stores Recent Gradients |
| Learning Rate Decreases Forever | Stable Learning Rate |
| Good for Sparse Data | Better General Purpose |
| Older Technique | Improved Technique |

---

# RMSProp vs Momentum

| Momentum | RMSProp |
|-----------|----------|
| Uses Gradient Direction | Uses Gradient Magnitude |
| EWMA of Gradients | EWMA of Squared Gradients |
| Faster Movement | Adaptive Learning Rate |

---

# RMSProp vs Adam

| RMSProp | Adam |
|----------|------|
| One EWMA | Two EWMAs |
| Simpler | More Powerful |
| Good Performance | Better Performance |
| Adaptive Learning Rate | Adaptive + Momentum |

---

# Relationship with Adam

Adam combines:

```text
Momentum
+
RMSProp
```

Specifically:

### Momentum Part



---

### RMSProp Part



Thus:

```text
Understanding RMSProp
is necessary before Adam.
```

---

# TensorFlow Example

```python
from tensorflow.keras.optimizers import RMSprop

optimizer = RMSprop(
    learning_rate=0.001,
    rho=0.9
)
```

---

# PyTorch Example

```python
import torch.optim as optim

optimizer = optim.RMSprop(
    model.parameters(),
    lr=0.001
)
```

---

# Industry Applications

RMSProp has been used in:

- Computer Vision
- Speech Recognition
- Reinforcement Learning
- Time Series Forecasting
- RNN Training

It remains an important optimizer in Deep Learning history.

---

# Important Terms

| Term | Meaning |
|---------|---------|
| RMSProp | Root Mean Square Propagation |
| EWMA | Exponentially Weighted Moving Average |
| Beta (β) | Decay Parameter |
| Adaptive Learning Rate | Parameter-specific learning rate |
| Squared Gradient | Gradient² |
| Adam | Momentum + RMSProp |

---

# Interview Questions

## Q1. What problem does RMSProp solve?

It prevents the learning rate from continuously shrinking as in AdaGrad.

---

## Q2. What does RMSProp stand for?

Root Mean Square Propagation.

---

## Q3. How does RMSProp differ from AdaGrad?

RMSProp uses a moving average of squared gradients instead of accumulating all gradients forever.

---

## Q4. What is the role of β in RMSProp?

It controls how much past gradient information is remembered.

---

## Q5. Which optimizer combines Momentum and RMSProp?

Adam.

---

# Key Points Learned Today

✅ Learned RMSProp Optimizer

✅ Understood AdaGrad's Limitation

✅ Learned EWMA of Squared Gradients

✅ Understood Adaptive Learning Rates

✅ Learned RMSProp Mathematics

✅ Compared RMSProp with AdaGrad

✅ Connected RMSProp to Adam

---

# Mini Summary

RMSProp (Root Mean Square Propagation) is an adaptive learning rate optimizer designed to overcome AdaGrad's continuously shrinking learning rates. By maintaining an exponentially weighted moving average of squared gradients, RMSProp provides stable and efficient optimization, making it one of the foundational optimizers that inspired Adam.

---
#d
# Progress

✅ Completed Day 33 - RMSProp Optimizer
