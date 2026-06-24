# Day 40 - Comparing CNN vs ANN

## Introduction

Artificial Neural Networks (ANNs) and Convolutional Neural Networks (CNNs) are two of the most important architectures in Deep Learning.

Although CNNs are actually a specialized type of ANN, they are designed for different purposes.

Understanding the difference between CNN and ANN is one of the most frequently asked interview questions in:

- Machine Learning
- Deep Learning
- Computer Vision
- Data Science

This chapter provides a complete comparison between ANN and CNN.

---

# What is ANN?

ANN stands for:

```text
Artificial Neural Network
```

It is inspired by biological neurons and consists of:

```text
Input Layer
      ↓
Hidden Layers
      ↓
Output Layer
```

ANN is primarily used for:

- Structured Data
- Tabular Data
- Numerical Data
- Classification
- Regression

---

# ANN Architecture

```text
Input
 ↓
Hidden Layer
 ↓
Hidden Layer
 ↓
Output
```

Every neuron is connected to:

```text
Every Neuron
```

in the next layer.

This is called:

```text
Fully Connected Architecture
```

---

# What is CNN?

CNN stands for:

```text
Convolutional Neural Network
```

CNN is specifically designed for:

```text
Image Processing
```

and Computer Vision tasks.

CNN automatically learns:

```text
Edges
↓
Textures
↓
Shapes
↓
Objects
```

from images.

---

# CNN Architecture

```text
Image
 ↓
Convolution
 ↓
ReLU
 ↓
Pooling
 ↓
Flatten
 ↓
Dense Layer
 ↓
Output
```

---

# Fundamental Difference

### ANN

Treats input as:

```text
Independent Features
```

---

### CNN

Treats input as:

```text
Spatial Data
```

and preserves relationships between neighboring pixels.

---

# Example

Suppose we have an image:

```text
224 × 224 × 3
```

---

# ANN View

ANN converts image into:

```text
150,528 Numbers
```

and processes them independently.

Result:

```text
Spatial Information Lost
```

---

# CNN View

CNN processes:

```text
Pixels + Their Positions
```

Result:

```text
Spatial Information Preserved
```

---

# Feature Extraction

### ANN

Feature extraction is often:

```text
Manual
```

---

### CNN

Feature extraction is:

```text
Automatic
```

through convolution layers.

---

# Learning Process

## ANN

Learns:

```text
Relationships Between Features
```

---

## CNN

Learns:

```text
Edges
↓
Corners
↓
Textures
↓
Shapes
↓
Objects
```

---

# Parameter Count

Consider:

```text
224 × 224 × 3 Image
```

---

# ANN

Input neurons:

```text
150,528
```

Connecting to:

```text
1000 Hidden Neurons
```

requires:

```text
150 Million+
Parameters
```

---

# CNN

Uses:

```text
Shared Filters
```

Therefore:

```text
Far Fewer Parameters
```

---

# Why CNN is Efficient?

CNN uses:

```text
Parameter Sharing
```

The same filter is reused across the image.

ANN cannot do this.

---

# Local Connectivity

## ANN

```text
Every Input
Connected Everywhere
```

---

## CNN

```text
Local Regions
Connected Locally
```

This greatly reduces computation.

---

# Translation Invariance

Suppose a cat moves slightly in an image.

---

### ANN

May struggle to recognize it.

---

### CNN

Still recognizes:

```text
Same Cat
Different Position
```

because of convolution and pooling.

---

# Memory Usage

### ANN

```text
Very High
```

for images.

---

### CNN

```text
Much Lower
```

due to shared filters.

---

# Training Speed

### ANN

Slow for image data.

---

### CNN

Much faster and more scalable.

---

# Accuracy on Images

### ANN

Lower accuracy.

---

### CNN

State-of-the-art performance.

---

# Layer Comparison

| ANN Layer | CNN Equivalent |
|------------|----------------|
| Dense Layer | Convolution Layer |
| Activation Layer | Activation Layer |
| Output Layer | Output Layer |

CNN includes additional layers:

- Convolution
- Pooling
- Flatten

which ANN lacks.

---

# Applications of ANN

ANN is commonly used for:

- House Price Prediction
- Stock Prediction
- Customer Churn Prediction
- Fraud Detection
- Credit Scoring
- Regression Problems

---

# Applications of CNN

CNN is commonly used for:

- Face Recognition
- Medical Imaging
- Object Detection
- Self-Driving Cars
- Satellite Image Analysis
- Image Classification

---

# Real World Example

### Customer Churn Prediction

Input:

```text
Age
Salary
Balance
Tenure
```

Best Choice:

```text
ANN
```

---

### Cat vs Dog Classification

Input:

```text
Image
```

Best Choice:

```text
CNN
```

---

# CNN vs ANN Comparison Table

| Feature | ANN | CNN |
|----------|------|------|
| Full Form | Artificial Neural Network | Convolutional Neural Network |
| Data Type | Structured Data | Images & Spatial Data |
| Feature Extraction | Manual | Automatic |
| Spatial Information | Lost | Preserved |
| Parameters | Very High | Lower |
| Memory Usage | High | Efficient |
| Translation Invariance | No | Yes |
| Accuracy on Images | Lower | Higher |
| Training Speed | Slower for Images | Faster |
| Computer Vision | Poor | Excellent |

---

# Advantages of ANN

✅ Simple Architecture

✅ Easy to Implement

✅ Works Well on Tabular Data

✅ Suitable for Regression Problems

---

# Disadvantages of ANN

❌ Too Many Parameters

❌ Poor Image Handling

❌ Spatial Information Loss

❌ Computationally Expensive

---

# Advantages of CNN

✅ Automatic Feature Extraction

✅ Fewer Parameters

✅ Better Accuracy

✅ Efficient Training

✅ Excellent for Images

---

# Disadvantages of CNN

❌ More Complex Architecture

❌ Requires More Data

❌ Higher Training Time than Small ANNs

---

# Industry Perspective

### Tabular Data

Most companies use:

```text
ANN
```

or traditional ML models.

---

### Image Data

Industry standard:

```text
CNN
```

Examples:

- Face Unlock
- Medical Diagnosis
- Autonomous Vehicles
- Security Cameras

---

# TensorFlow Example

## ANN

```python
model = Sequential()

model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
```

---

## CNN

```python
model = Sequential()

model.add(
    Conv2D(
        32,
        (3,3),
        activation='relu'
    )
)

model.add(MaxPooling2D((2,2)))

model.add(Flatten())

model.add(Dense(10, activation='softmax'))
```

---

# Interview Questions

## Q1. What is the main difference between ANN and CNN?

ANN processes features independently, while CNN preserves spatial relationships using convolution.

---

## Q2. Why is CNN preferred for images?

Because it automatically extracts features and preserves spatial information.

---

## Q3. Why does CNN use fewer parameters?

Because filters are shared across the image.

---

## Q4. Which is better for customer churn prediction?

ANN.

---

## Q5. Which is better for image classification?

CNN.

---

# Key Points Learned Today

✅ Learned ANN Architecture

✅ Learned CNN Architecture

✅ Compared ANN vs CNN

✅ Understood Parameter Sharing

✅ Learned Spatial Information Preservation

✅ Compared Real-World Applications

✅ Studied Industry Usage

---
#t
# Mini Summary

ANNs are general-purpose neural networks best suited for structured and tabular data, while CNNs are specialized neural networks designed for image and spatial data. CNNs preserve spatial relationships, use parameter sharing, automatically extract features, and achieve significantly better performance on computer vision tasks compared to traditional ANNs.

---

# Progress

✅ Completed Day 40 - Comparing CNN vs ANN
