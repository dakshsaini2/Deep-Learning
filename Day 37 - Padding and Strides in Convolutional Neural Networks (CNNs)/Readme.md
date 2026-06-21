# Day 37 - Padding and Strides in Convolutional Neural Networks (CNNs)

## Introduction

In the previous chapter, we learned about:

- CNN Architecture
- Visual Cortex Inspiration
- Convolution Layers

When a filter (kernel) moves over an image, two important concepts determine how the convolution operation works:

```text
1. Padding
2. Stride
```

These concepts directly affect:

- Output Size
- Information Preservation
- Computational Cost
- Feature Extraction

Understanding Padding and Stride is essential for designing CNN architectures.

---

# Recap: Convolution Operation

A filter slides over an image and performs:

```text
Element-wise Multiplication
+
Summation
```

Example:

Input Image

```text
5 × 5
```

Kernel

```text
3 × 3
```

Output:

```text
Feature Map
```

---

# Problem Without Padding

Suppose:

Input Image:

```text
5 × 5
```

Kernel:

```text
3 × 3
```

Output Size:

```text
3 × 3
```

Notice:

```text
5 × 5
↓
3 × 3
```

Image size decreases.

---

# Why Is This a Problem?

After multiple convolution layers:

```text
224 × 224
↓
222 × 222
↓
220 × 220
↓
...
```

The feature map becomes very small.

Important information may be lost.

---

# What is Padding?

Padding means:

```text
Adding Extra Pixels
Around Image Borders
```

before convolution.

Usually:

```text
Zeros
```

are added.

Therefore it is also called:

```text
Zero Padding
```

---

# Example of Padding

Original Image:

```text
1 2 3
4 5 6
7 8 9
```

Padding = 1

Result:

```text
0 0 0 0 0
0 1 2 3 0
0 4 5 6 0
0 7 8 9 0
0 0 0 0 0
```

Image becomes:

```text
5 × 5
```

---

# Why Use Padding?

Padding helps:

✅ Preserve Image Size

✅ Preserve Border Information

✅ Enable Deep CNNs

✅ Improve Feature Learning

---

# Types of Padding

## 1. Valid Padding

No padding applied.

```text
Padding = 0
```

Output size decreases.

---

## 2. Same Padding

Padding added such that:

```text
Output Size
=
Input Size
```

Most commonly used in modern CNNs.

---

# Valid Padding Example

Input:

```text
5 × 5
```

Kernel:

```text
3 × 3
```

Padding:

```text
0
```

Output:

```text
3 × 3
```

---

# Same Padding Example

Input:

```text
5 × 5
```

Kernel:

```text
3 × 3
```

Padding:

```text
1
```

Output:

```text
5 × 5
```

---

# Output Size Formula

For Convolution:



Where:

- N = Input Size
- F = Filter Size
- P = Padding
- S = Stride

---

# Example Calculation

Input:

```text
N = 5
```

Filter:

```text
F = 3
```

Padding:

```text
P = 1
```

Stride:

```text
S = 1
```

Output:



Output:

```text
5 × 5
```

---

# What is Stride?

Stride determines:

```text
How Many Pixels
The Filter Moves
```

during convolution.

---

# Stride = 1

Filter moves:

```text
One Pixel At A Time
```

Example:

```text
□ □ □
□ □ □
□ □ □
```

↓

```text
Move 1 Step →
```

---

# Stride = 2

Filter moves:

```text
Two Pixels At A Time
```

Example:

```text
Move 2 Steps →
```

Output becomes smaller.

---

# Effect of Stride

Larger stride:

```text
Smaller Feature Maps
```

Smaller stride:

```text
Larger Feature Maps
```

---

# Example

Input:

```text
7 × 7
```

Kernel:

```text
3 × 3
```

Padding:

```text
0
```

Stride:

```text
1
```

Output:

```text
5 × 5
```

---

Now:

Stride:

```text
2
```

Output:

```text
3 × 3
```

Much smaller.

---

# Why Use Stride?

Benefits:

✅ Reduces Computation

✅ Reduces Memory Usage

✅ Faster Training

---

# Trade-Off

Small Stride:

```text
More Information
More Computation
```

---

Large Stride:

```text
Less Information
Less Computation
```

---

# Padding vs Stride

| Padding | Stride |
|----------|---------|
| Adds Pixels | Moves Filter |
| Preserves Size | Reduces Size |
| Retains Border Information | Controls Downsampling |

---

# Visualization

## Padding

```text
Before

1 2 3
4 5 6
7 8 9

After

0 0 0 0 0
0 1 2 3 0
0 4 5 6 0
0 7 8 9 0
0 0 0 0 0
```

---

## Stride

Stride = 1

```text
Move:
→ → → →
```

---

Stride = 2

```text
Move:
→→ →→
```

---

# Common Industry Settings

Most CNNs use:

```text
Kernel = 3×3

Stride = 1

Padding = Same
```

Examples:

- VGG16
- ResNet
- EfficientNet

---

# TensorFlow Example

```python
Conv2D(
    filters=32,
    kernel_size=(3,3),
    strides=(1,1),
    padding='same'
)
```

---

# PyTorch Example

```python
nn.Conv2d(
    in_channels=3,
    out_channels=32,
    kernel_size=3,
    stride=1,
    padding=1
)
```

---

# Real World Applications

Padding and Stride are used in:

- Image Classification
- Object Detection
- Face Recognition
- Medical Imaging
- Autonomous Vehicles

Every CNN architecture uses these concepts.

---

# Important Terms

| Term | Meaning |
|---------|---------|
| Padding | Adding extra pixels around borders |
| Zero Padding | Padding using zeros |
| Valid Padding | No padding |
| Same Padding | Output size equals input size |
| Stride | Filter movement step size |
| Feature Map | Output of convolution |

---

# Interview Questions

## Q1. Why is padding used?

To preserve image dimensions and border information.

---

## Q2. What is zero padding?

Adding zeros around image borders before convolution.

---

## Q3. What happens when stride increases?

Output feature map size decreases.

---

## Q4. What is same padding?

Padding that keeps output size equal to input size.

---

## Q5. What is the output size formula for convolution?



---

# Key Points Learned Today

✅ Learned Padding

✅ Understood Zero Padding

✅ Learned Valid and Same Padding

✅ Learned Stride

✅ Understood Output Size Calculation

✅ Compared Padding vs Stride

✅ Studied Industry Best Practices

---

# Mini Summary

Padding and Stride are fundamental concepts in CNNs. Padding preserves image dimensions and border information by adding pixels around the image, while Stride controls how far a filter moves during convolution. Together, they determine feature map size, computational efficiency, and the amount of information retained during learning.

---

# Progress

✅ Completed Day 37 - Padding and Strides in CNNs
