# Day 46 - What is Transfer Learning? | Transfer Learning in Keras

## Introduction

Training a Deep Learning model from scratch requires:

- Millions of labeled images
- High-end GPUs or TPUs
- Large amounts of memory
- Significant training time

For most real-world projects, collecting such large datasets is impractical.

Instead of starting with random weights, we can reuse the knowledge learned by an already trained model.

This approach is called:

```text
Transfer Learning
```

Transfer Learning is one of the most widely used techniques in modern Computer Vision and is supported directly in **Keras** through pre-trained models such as:

- VGG16
- VGG19
- ResNet50
- InceptionV3
- Xception
- MobileNetV2
- EfficientNet

---

# What is Transfer Learning?

Transfer Learning is the process of:

```text
Using a Model
Trained on One Task

↓

Applying It
to Another Related Task
```

Instead of learning from scratch, the model starts with knowledge already learned from a large dataset.

---

# Simple Analogy

Imagine a person who already knows:

```text
English
```

Learning:

```text
Spanish
```

becomes easier.

Similarly,

A CNN trained on millions of images already knows:

- Edges
- Textures
- Shapes
- Objects

We simply teach it a new task.

---

# Traditional Deep Learning

```text
Dataset

↓

Initialize Random Weights

↓

Train CNN

↓

Prediction
```

Training may take days or weeks.

---

# Transfer Learning

```text
Pre-trained CNN

↓

Replace Output Layer

↓

Fine Tune

↓

Prediction
```

Training becomes much faster.

---

# Why Does Transfer Learning Work?

Large CNNs are usually trained on:

```text
ImageNet
```

ImageNet contains:

```text
14 Million+

Images

1000 Classes
```

The model already understands:

- Animals
- Cars
- Flowers
- Humans
- Buildings
- Nature

Therefore:

```text
It Does Not Need
to Learn Basic Features Again
```

---

# Feature Learning in CNN

Early Layers Learn:

```text
Edges

Textures

Lines
```

---

Middle Layers Learn:

```text
Shapes

Patterns

Parts
```

---

Deep Layers Learn:

```text
Objects

Faces

Animals
```

These features are useful for many vision tasks.

---

# Types of Transfer Learning

There are two major approaches.

---

# 1. Feature Extraction

Freeze the entire pre-trained CNN.

Train only the final classifier.

```text
CNN

↓

Frozen

↓

New Dense Layer

↓

Prediction
```

Best for:

- Small datasets
- Fast training

---

# 2. Fine-Tuning

Train the last few convolution layers along with the classifier.

```text
CNN

↓

Last Layers Trainable

↓

Classifier

↓

Prediction
```

Best for:

- Medium-sized datasets
- Higher accuracy

---

# Feature Extraction vs Fine-Tuning

| Feature Extraction | Fine-Tuning |
|--------------------|------------|
| Freeze CNN | Train Some CNN Layers |
| Faster | Slower |
| Less Data Required | More Data Required |
| Lower GPU Usage | Higher GPU Usage |
| Good Accuracy | Better Accuracy |

---

# Popular Pre-trained Models in Keras

| Model | Input Size | Parameters | Applications |
|--------|------------|-----------:|--------------|
| VGG16 | 224×224 | 138 Million | General Computer Vision |
| ResNet50 | 224×224 | 25 Million | Image Classification |
| MobileNetV2 | 224×224 | 3.5 Million | Mobile Devices |
| EfficientNetB0 | 224×224 | 5.3 Million | High Accuracy |
| Xception | 299×299 | 23 Million | Advanced Vision Tasks |
| InceptionV3 | 299×299 | 24 Million | Image Recognition |

---

# Transfer Learning Workflow

```text
Collect Dataset
        ↓
Preprocess Images
        ↓
Load Pre-trained Model
        ↓
Freeze Layers
        ↓
Add New Classifier
        ↓
Train Model
        ↓
Evaluate Model
```

---

# Loading a Pre-trained Model in Keras

Example using ResNet50:

```python
from tensorflow.keras.applications import ResNet50

base_model = ResNet50(

    weights='imagenet',

    include_top=False,

    input_shape=(224,224,3)

)
```

Explanation:

- `weights='imagenet'` loads weights learned from the ImageNet dataset.
- `include_top=False` removes the original classification head.
- `input_shape` defines the size of your input images.

---

# Freezing the Base Model

Freeze all convolution layers:

```python
base_model.trainable = False
```

Now only the newly added layers will learn.

---

# Adding Custom Layers

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout

model = Sequential([

    base_model,

    GlobalAveragePooling2D(),

    Dense(256, activation='relu'),

    Dropout(0.5),

    Dense(1, activation='sigmoid')

])
```

---

# Why GlobalAveragePooling2D?

Instead of using:

```text
Flatten
```

modern CNNs often use:

```text
GlobalAveragePooling2D
```

Benefits:

✅ Fewer Parameters

✅ Less Overfitting

✅ Faster Training

---

# Model Compilation

```python
model.compile(

optimizer='adam',

loss='binary_crossentropy',

metrics=['accuracy']

)
```

---

# Training the Model

```python
history = model.fit(

train_generator,

validation_data=validation_generator,

epochs=10

)
```

---

# Fine-Tuning Example

After initial training:

```python
base_model.trainable = True
```

Freeze only early layers:

```python
for layer in base_model.layers[:-30]:
    layer.trainable = False
```

Train the remaining layers:

```python
model.compile(

optimizer=tf.keras.optimizers.Adam(1e-5),

loss='binary_crossentropy',

metrics=['accuracy']

)
```

A smaller learning rate is recommended during fine-tuning to avoid large updates to the pre-trained weights.

---

# Why Use a Smaller Learning Rate?

The model already has good learned features.

Large updates may:

❌ Destroy useful knowledge.

Small learning rates:

✅ Preserve learned features.

✅ Improve fine-tuning.

---

# Example Project

Cat vs Dog Classification

```text
Cat Images

↓

ResNet50

↓

GlobalAveragePooling

↓

Dense Layer

↓

Sigmoid

↓

Prediction
```

---

# Advantages of Transfer Learning

✅ Faster Training

✅ Higher Accuracy

✅ Less Data Required

✅ Reduced Overfitting

✅ Lower Computational Cost

---

# Limitations

❌ Performance may drop if the source and target tasks are very different.

❌ Large models require more memory.

❌ Fine-tuning too many layers on a very small dataset can cause overfitting.

---

# Real-World Applications

Transfer Learning is used in:

- Medical Image Classification
- Face Recognition
- Defect Detection
- Plant Disease Detection
- Wildlife Monitoring
- Self-Driving Cars
- Satellite Image Analysis

---

# Industry Perspective

Most companies do **not** train CNNs from scratch.

Instead they use:

- ResNet
- EfficientNet
- MobileNet
- Xception
- Vision Transformers (ViT)

and fine-tune them for specific business problems.

---

# Best Practices

✅ Use ImageNet pre-trained weights whenever possible.

✅ Freeze the base model for small datasets.

✅ Fine-tune only higher layers if more data is available.

✅ Use data augmentation with transfer learning.

✅ Reduce the learning rate during fine-tuning.

---

# Important Terms

| Term | Meaning |
|------|---------|
| Transfer Learning | Reusing knowledge from a pre-trained model |
| Pre-trained Model | Model already trained on a large dataset |
| Feature Extraction | Freeze the CNN and train only the classifier |
| Fine-Tuning | Train some pre-trained layers along with the classifier |
| ImageNet | Large image dataset with over 14 million images and 1,000 classes |
| GlobalAveragePooling2D | Reduces feature maps by averaging them |

---

# Interview Questions

## Q1. What is Transfer Learning?

Transfer Learning is the process of reusing a model trained on one task to solve another related task.

---

## Q2. Why is Transfer Learning better than training from scratch?

It requires less data, trains faster, and often achieves better accuracy.

---

## Q3. What is the difference between Feature Extraction and Fine-Tuning?

Feature Extraction freezes the pre-trained model and trains only the new classifier, while Fine-Tuning also updates some of the pre-trained layers.

---

## Q4. Why do we use `include_top=False` in Keras?

To remove the original classification layer so we can add a custom classifier for a new task.

---

## Q5. Why is `GlobalAveragePooling2D` preferred over `Flatten`?

It reduces the number of parameters, lowers the risk of overfitting, and is more computationally efficient.

---

# Key Points Learned Today

✅ Learned Transfer Learning

✅ Understood Pre-trained Models

✅ Learned Feature Extraction

✅ Understood Fine-Tuning

✅ Learned Transfer Learning in Keras

✅ Studied ResNet50 Implementation

✅ Learned Industry Best Practices

---

# Mini Summary

Transfer Learning is one of the most powerful techniques in modern Deep Learning. Instead of training a CNN from scratch, it leverages a model that has already learned rich visual features from a large dataset such as ImageNet. In Keras, pre-trained models like ResNet50, VGG16, MobileNetV2, and EfficientNet can be loaded with a few lines of code. By freezing the convolutional base and training a custom classifier—or fine-tuning the upper layers—you can build accurate computer vision models with significantly less data and training time.

---

# Progress

✅ Completed Day 46 - What is Transfer Learning? | Transfer Learning in Keras
