# Day 45 - What Does a CNN See? | Visualizing CNN Filters and Feature Maps

## Introduction

One of the biggest questions in Deep Learning is:

```text
"What does a CNN actually learn?"
```

Unlike humans, a CNN does **not** see an image as:

- Cat
- Dog
- Car
- Person

Initially, a CNN only sees:

```text
Numbers (Pixel Values)
```

Through training, CNN gradually learns meaningful visual patterns such as:

```text
Edges
↓

Corners
↓

Textures
↓

Shapes
↓

Object Parts
↓

Complete Objects
```

This process can be understood using **CNN Visualization** techniques.

CNN visualization helps us look inside a neural network and understand what each layer has learned.

---

# What Does a CNN Actually See?

Suppose we give a CNN this image:

```text
🐱 Cat
```

Humans immediately recognize:

```text
Cat
```

CNN sees:

```text
Pixel Matrix
```

Example:

```text
[125 132 140 ...]

[110 118 127 ...]

[95 102 108 ...]
```

Initially:

```text
No Meaning
```

Only numerical values.

---

# How Does CNN Learn?

During training:

```text
Image
↓

Convolution

↓

Feature Maps

↓

Pooling

↓

More Convolution

↓

Prediction
```

Each layer learns more complex information.

---

# Hierarchical Feature Learning

CNN learns features in stages.

---

# Layer 1

Learns:

```text
Edges

Lines

Simple Patterns
```

Example:

```text
────

│

╱

╲
```

---

# Layer 2

Combines edges into:

```text
Corners

Curves

Textures
```

---

# Layer 3

Learns:

```text
Eyes

Nose

Fur

Ears
```

---

# Deep Layers

Recognize:

```text
Entire Cat
```

or

```text
Entire Dog
```

---

# CNN Learning Hierarchy

```text
Pixels

↓

Edges

↓

Corners

↓

Textures

↓

Shapes

↓

Object Parts

↓

Objects
```

---

# What is a Filter?

A filter (kernel) is a small learnable matrix.

Example:

```text
3 × 3 Filter
```

Initially:

```text
Random Numbers
```

Example:

```text
0.12

-0.45

0.31
```

During training:

The filter becomes specialized.

---

# Example Filters

### Vertical Edge Filter

```text
1 0 -1

1 0 -1

1 0 -1
```

---

### Horizontal Edge Filter

```text
1 1 1

0 0 0

-1 -1 -1
```

---

### Blur Filter

```text
1 1 1

1 1 1

1 1 1
```

---

# What is a Feature Map?

A feature map is:

```text
Output of a Filter
```

Example:

```text
Image

×

Edge Filter

↓

Feature Map
```

The feature map highlights:

```text
Only Edges
```

---

# Multiple Filters

CNN does not use:

```text
One Filter
```

It learns:

```text
32

64

128

256

Filters
```

Each detects something different.

---

# Example

Filter 1

```text
Vertical Edges
```

Filter 2

```text
Horizontal Edges
```

Filter 3

```text
Curves
```

Filter 4

```text
Textures
```

---

# What Does Layer 1 Learn?

Mostly:

```text
Edges
```

Visualization:

```text
████░░░░

░░██████

██░░████
```

High values correspond to detected edges.

---

# What Does Layer 2 Learn?

Combines edges into:

```text
Corners

Textures

Small Patterns
```

---

# What Do Middle Layers Learn?

Middle layers detect:

```text
Eyes

Fur

Legs

Whiskers
```

---

# Deep Layers

Final convolution layers detect:

```text
Entire Face

Entire Animal

Entire Object
```

---

# Why Visualize CNN?

Visualization helps us:

✅ Understand model behavior

✅ Detect overfitting

✅ Explain predictions

✅ Debug CNNs

✅ Build trustworthy AI

---

# Visualization Techniques

## 1. Filter Visualization

Shows:

```text
Learned Filters
```

---

## 2. Feature Map Visualization

Shows:

```text
Activation Maps
```

after convolution.

---

## 3. Activation Maximization

Generates an image that:

```text
Maximally Activates
One Neuron
```

---

## 4. Saliency Maps

Highlights:

```text
Most Important Pixels
```

for a prediction.

---

## 5. Grad-CAM

Produces:

```text
Heatmaps
```

showing where the CNN focused.

Example:

```text
Dog

⬤⬤⬤

Head Highlighted
```

instead of background.

---

# Grad-CAM Example

Input:

```text
Golden Retriever
```

Heatmap:

```text
Face

Eyes

Nose
```

Bright colors indicate important regions.

---

# CNN Visualization Pipeline

```text
Input Image

↓

Convolution

↓

Feature Maps

↓

Visualization

↓

Interpretation
```

---

# Why Filters Change During Training?

Initially:

```text
Random Filters
```

After backpropagation:

```text
Meaningful Filters
```

that detect important visual features.

---

# Example Evolution

Epoch 1

```text
Random Noise
```

↓

Epoch 10

```text
Edges
```

↓

Epoch 50

```text
Textures
```

↓

Epoch 100

```text
Faces
```

---

# TensorFlow Example

```python
from tensorflow.keras.models import Model

layer_outputs = [
    layer.output
    for layer in model.layers[:8]
]

activation_model = Model(
    inputs=model.input,
    outputs=layer_outputs
)

activations = activation_model.predict(image)
```

This extracts feature maps from the first eight layers.

---

# Visualizing Filters

```python
filters, biases = model.layers[0].get_weights()

print(filters.shape)
```

Output:

```text
(3, 3, 3, 32)
```

Meaning:

- Filter Size = 3 × 3
- RGB Channels = 3
- Number of Filters = 32

---

# Applications

CNN visualization is used in:

- Medical AI
- Autonomous Vehicles
- Face Recognition
- Explainable AI (XAI)
- Model Debugging
- AI Safety

---

# Advantages

✅ Makes CNN decisions interpretable

✅ Helps improve model performance

✅ Detects incorrect learning

✅ Builds trust in AI systems

---

# Limitations

❌ Early filters are easy to interpret.

❌ Deep filters become abstract.

❌ Visualization techniques may not perfectly explain every decision.

---

# Important Terms

| Term | Meaning |
|---------|---------|
| Filter (Kernel) | Learnable matrix that extracts features |
| Feature Map | Output produced by applying a filter |
| Activation Map | Values showing neuron responses |
| Saliency Map | Highlights important pixels |
| Grad-CAM | Heatmap showing important image regions |
| Explainable AI (XAI) | Techniques that make AI decisions understandable |

---

# Interview Questions

## Q1. What does a CNN see initially?

Only pixel values represented as numerical matrices.

---

## Q2. What is a filter in CNN?

A learnable kernel that extracts specific features such as edges or textures.

---

## Q3. What is a feature map?

The output generated after applying a filter to an input image.

---

## Q4. What does Grad-CAM show?

It highlights the image regions that contributed the most to the model's prediction.

---

## Q5. Why is CNN visualization important?

It improves interpretability, debugging, and trust in deep learning models.

---

# Key Points Learned Today

✅ Learned what a CNN "sees"

✅ Understood Filters and Kernels

✅ Learned Feature Maps

✅ Understood Hierarchical Feature Learning

✅ Learned CNN Visualization Techniques

✅ Studied Grad-CAM and Saliency Maps

✅ Understood Explainable AI (XAI)

---

# Mini Summary

A CNN does not recognize objects directly. It begins by processing images as matrices of pixel values and gradually learns hierarchical visual features through convolution. Early layers detect edges and simple patterns, intermediate layers identify textures and object parts, and deeper layers recognize complete objects. Visualization techniques such as feature maps, filter visualization, saliency maps, and Grad-CAM allow us to understand what the network has learned, making CNNs more interpretable and trustworthy.

---

# Progress

✅ Completed Day 45 - What Does a CNN See? | Visualizing CNN Filters and Feature Maps
