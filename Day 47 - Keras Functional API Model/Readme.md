# Day 47 - Keras Functional API Model

## Introduction

Until now, most of our Deep Learning models have been built using:

```python
Sequential()
```

The Sequential API is simple and suitable for linear neural networks.

However, modern Deep Learning architectures such as:

- ResNet
- Inception
- DenseNet
- U-Net
- Siamese Networks
- Multi-Input Models
- Multi-Output Models

cannot be built efficiently using the Sequential API.

To build these advanced architectures, Keras provides:

```text
Functional API
```

The Functional API allows us to create flexible and complex neural networks by defining the flow of data between layers.

It is the preferred approach for building production-grade Deep Learning models.

---

# What is the Keras Functional API?

The Functional API is a way of building neural networks where:

```text
Layers are Connected
Like a Directed Graph
```

instead of stacking them one after another.

Unlike the Sequential API:

```text
Input

↓

Layer 1

↓

Layer 2

↓

Output
```

the Functional API allows:

- Multiple Inputs
- Multiple Outputs
- Branching
- Skip Connections
- Shared Layers

---

# Why Do We Need Functional API?

Sequential models work only when:

```text
One Input

↓

One Output

↓

No Branching
```

But real-world models are often much more complex.

Examples:

```text
Image + Text

↓

One Prediction
```

or

```text
One Image

↓

Two Different Outputs
```

These architectures require the Functional API.

---

# Sequential vs Functional API

## Sequential API

```text
Input

↓

Dense

↓

Dense

↓

Output
```

Simple linear architecture.

---

## Functional API

```text
          Dense
         /

Input

         \

          Dense

            ↓

      Concatenate

            ↓

         Output
```

Supports branching and merging.

---

# Advantages of Functional API

✅ Multiple Inputs

✅ Multiple Outputs

✅ Shared Layers

✅ Skip Connections

✅ Residual Networks

✅ Flexible Architecture

---

# Functional API Workflow

```text
Input Layer

↓

Hidden Layers

↓

Output Layer

↓

Model Creation
```

---

# Step 1 : Import Libraries

```python
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
```

---

# Step 2 : Create Input Layer

```python
inputs = Input(shape=(10,))
```

Explanation:

```text
10 Input Features
```

Unlike Sequential:

The input layer is created separately.

---

# Step 3 : Hidden Layers

```python
x = Dense(

64,

activation='relu'

)(inputs)
```

Notice:

```python
(inputs)
```

Layers behave like Python functions.

---

Second Layer:

```python
x = Dense(

32,

activation='relu'

)(x)
```

---

# Step 4 : Output Layer

Binary Classification:

```python
outputs = Dense(

1,

activation='sigmoid'

)(x)
```

---

# Step 5 : Create Model

```python
model = Model(

inputs=inputs,

outputs=outputs

)
```

This connects:

```text
Input

↓

Hidden Layers

↓

Output
```

into one complete model.

---

# Step 6 : Compile Model

```python
model.compile(

optimizer='adam',

loss='binary_crossentropy',

metrics=['accuracy']

)
```

---

# Step 7 : Train Model

```python
model.fit(

X_train,

y_train,

epochs=20,

batch_size=32

)
```

---

# Complete Functional API Example

```python
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model

inputs = Input(shape=(10,))

x = Dense(

64,

activation='relu'

)(inputs)

x = Dense(

32,

activation='relu'

)(x)

outputs = Dense(

1,

activation='sigmoid'

)(x)

model = Model(

inputs,

outputs

)

model.compile(

optimizer='adam',

loss='binary_crossentropy',

metrics=['accuracy']

)

model.summary()
```

---

# Multi-Input Model

Suppose we have:

```text
Image Data

+

Age Data
```

Both can be processed together.

Architecture:

```text
Image Input

↓

CNN

↓

Feature Vector

             \

              Concatenate

             /

Age Input

↓

Dense

↓

Output
```

Example:

```python
image_input = Input(shape=(224,224,3))

age_input = Input(shape=(1,))
```

---

# Multi-Output Model

Suppose one model predicts:

- Age
- Gender

Architecture:

```text
Input

↓

Shared Layers

↓

Age Output

Gender Output
```

Example:

```python
age_output = Dense(1)(x)

gender_output = Dense(1, activation='sigmoid')(x)
```

---

# Shared Layers

One layer can be reused.

Example:

```python
shared = Dense(

128,

activation='relu'

)

output1 = shared(input1)

output2 = shared(input2)
```

Applications:

- Siamese Networks
- Face Verification
- Signature Matching

---

# Skip Connections

Used in:

```text
ResNet
```

Architecture:

```text
Input

↓

Conv

↓

Conv

↓

Add Input

↓

Output
```

This helps train:

```text
Very Deep Networks
```

---

# Functional API vs Sequential API

| Sequential API | Functional API |
|----------------|----------------|
| Linear Models Only | Complex Models |
| One Input | Multiple Inputs |
| One Output | Multiple Outputs |
| No Skip Connections | Supports Skip Connections |
| Easy to Learn | More Flexible |
| Good for Beginners | Industry Standard |

---

# Functional API in CNN

Example:

```python
inputs = Input(shape=(224,224,3))

x = Conv2D(

32,

(3,3),

activation='relu'

)(inputs)

x = MaxPooling2D()(x)

x = Flatten()(x)

outputs = Dense(

10,

activation='softmax'

)(x)

model = Model(inputs, outputs)
```

---

# Model Summary

```python
model.summary()
```

Displays:

- Layers
- Parameters
- Output Shapes

---

# Model Visualization

```python
from tensorflow.keras.utils import plot_model

plot_model(

model,

show_shapes=True,

show_layer_names=True

)
```

This generates a diagram of the network architecture.

---

# Real-World Applications

The Functional API is used in:

- ResNet
- Inception
- U-Net
- DenseNet
- MobileNet
- Multi-modal AI
- Medical Image Analysis
- Autonomous Vehicles

---

# Best Practices

✅ Use Sequential API for simple linear models.

✅ Use Functional API for complex architectures.

✅ Name important layers for readability.

✅ Visualize the model using `plot_model()`.

---

# Important Terms

| Term | Meaning |
|---------|---------|
| Functional API | Flexible way to build neural networks |
| Input Layer | Starting point of the model |
| Model | Connects inputs and outputs |
| Multi-Input Model | Model with more than one input |
| Multi-Output Model | Model with multiple predictions |
| Shared Layer | Same layer reused in different paths |
| Skip Connection | Direct connection between non-adjacent layers |

---

# Interview Questions

## Q1. What is the Keras Functional API?

A flexible API for building neural networks with complex architectures such as multiple inputs, multiple outputs, and skip connections.

---

## Q2. What is the difference between Sequential API and Functional API?

Sequential API builds linear models, whereas the Functional API supports complex directed graph architectures.

---

## Q3. Which popular architectures use the Functional API?

ResNet, Inception, DenseNet, U-Net, and Siamese Networks.

---

## Q4. Why is the Functional API preferred in industry?

Because it allows developers to build scalable and flexible architectures used in modern AI applications.

---

## Q5. Can the Functional API create models with multiple inputs and outputs?

Yes. This is one of its primary advantages over the Sequential API.

---

# Key Points Learned Today

✅ Learned Keras Functional API

✅ Understood Why It Is Needed

✅ Built a Functional Model

✅ Learned Multi-Input Networks

✅ Learned Multi-Output Networks

✅ Understood Shared Layers

✅ Learned Skip Connections

---

# Mini Summary

The Keras Functional API is a powerful and flexible way to build Deep Learning models that go beyond simple sequential architectures. It allows developers to design neural networks as directed graphs, enabling multiple inputs, multiple outputs, shared layers, and skip connections. Modern architectures such as ResNet, Inception, DenseNet, and U-Net are all implemented using the Functional API, making it an essential tool for building production-ready AI systems.

---
#a
# Progress

✅ Completed Day 47 - Keras Functional API Model
