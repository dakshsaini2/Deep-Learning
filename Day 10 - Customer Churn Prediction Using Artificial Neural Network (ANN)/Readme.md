# Day 10 - Customer Churn Prediction Using Artificial Neural Network (ANN)

## Introduction

Customer Churn Prediction is one of the most important real-world applications of Machine Learning and Deep Learning.

Companies use AI systems to predict:
- which customers may leave,
- subscription cancellations,
- user disengagement,
- revenue loss risks.

This helps businesses:
- improve retention,
- increase profits,
- optimize marketing strategies.

In this project, we use:
- Artificial Neural Networks (ANNs)
to predict customer churn.

---

# What is Customer Churn?

Customer Churn means:
- customers stopping use of a product or service.

Examples:
- Cancelling Netflix subscription
- Leaving a telecom provider
- Uninstalling an app
- Switching banks

---

# Why Churn Prediction is Important?

Acquiring new customers is expensive.

Companies prefer:
- retaining existing customers.

AI-powered churn prediction helps:
- identify risky customers early,
- take preventive actions.

---

# Real World Industry Usage

Customer churn prediction is widely used in:
- Banking
- Telecom
- SaaS Platforms
- E-commerce
- Streaming Platforms

Companies using churn prediction:
- :contentReference[oaicite:0]{index=0}
- :contentReference[oaicite:1]{index=1}
- :contentReference[oaicite:2]{index=2}
- :contentReference[oaicite:3]{index=3}

---

# Problem Statement

Goal:
- Predict whether a customer will churn or not.

This is a:
- Binary Classification Problem.

Output:
- 0 → Customer stays
- 1 → Customer leaves

---

# Why Use ANN?

Artificial Neural Networks are powerful because they:
- learn complex relationships,
- detect hidden patterns,
- handle non-linear data,
- scale well with large datasets.

ANNs often outperform traditional ML models on large structured datasets.

---

# Machine Learning Pipeline

```text
Data Collection
      ↓
Data Preprocessing
      ↓
Feature Engineering
      ↓
Train-Test Split
      ↓
Build ANN Model
      ↓
Training
      ↓
Evaluation
      ↓
Prediction
```

---

# Dataset Features

Example customer features:

| Feature | Description |
|---|---|
| Credit Score | Customer credit rating |
| Geography | Customer location |
| Gender | Male/Female |
| Age | Customer age |
| Balance | Bank balance |
| Tenure | Years with company |
| Products | Number of products used |
| Active Member | Customer activity |
| Estimated Salary | Customer income |

Target Variable:
- Exited (Churn)

---

# ANN Architecture

```text
Input Layer → Hidden Layer(s) → Output Layer
```

---

# ANN Architecture Diagram



---

# Input Layer

The input layer receives customer features.

Example:

:contentReference[oaicite:5]{index=5}

Where:
- each feature represents customer information.

---

# Hidden Layers

Hidden layers learn:
- customer behavior patterns,
- hidden relationships,
- risk indicators.

The network automatically learns:
- complex feature interactions.

---

# Output Layer

Since churn prediction is binary classification:
- output layer contains one neuron.

Prediction:
- close to 1 → likely to churn
- close to 0 → likely to stay

---

# Forward Propagation

Each layer computes:

:contentReference[oaicite:6]{index=6}

Then activation:

:contentReference[oaicite:7]{index=7}

---

# Activation Functions

## Hidden Layers → ReLU

:contentReference[oaicite:8]{index=8}

Advantages:
- fast,
- efficient,
- reduces vanishing gradients.

---

# Output Layer → Sigmoid

:contentReference[oaicite:9]{index=9}

Why sigmoid?
- produces probability between 0 and 1.

---

# Loss Function

For binary classification we use:
- Binary Cross Entropy (BCE).

Formula:

:contentReference[oaicite:10]{index=10}

Goal:
- minimize prediction error.

---

# Training Process

Training involves:
1. Forward Propagation
2. Loss Calculation
3. Backpropagation
4. Weight Updates

Repeated across multiple:
- epochs.

---

# Data Preprocessing

Real-world datasets require preprocessing.

---

# Common Preprocessing Steps

## 1. Handling Missing Values

Missing data must be:
- removed,
OR
- filled.

---

# 2. Encoding Categorical Variables

Convert:
- Male/Female
- Geography

into numerical format.

Techniques:
- Label Encoding
- One-Hot Encoding

---

# 3. Feature Scaling

Neural networks perform better when features are scaled.

Common techniques:
- Standardization
- Normalization

---

# Train-Test Split

Dataset is divided into:
- Training Set
- Testing Set

Purpose:
- evaluate generalization performance.

Typical split:
- 80% training
- 20% testing

---

# Why Deep Learning Works Well Here?

ANNs learn:
- hidden customer patterns,
- behavioral relationships,
- churn risk indicators.

Example:
- low activity + high balance + age factor
may indicate:
- high churn probability.

---

# Evaluation Metrics

## Accuracy

Measures:
- overall correctness.

---

# Precision

Measures:
- correctness of positive predictions.

---

# Recall

Measures:
- ability to detect actual churners.

---

# F1 Score

Balances:
- precision,
- recall.

---

# Confusion Matrix

Used to analyze:
- true positives,
- false positives,
- false negatives,
- true negatives.

---

# Churn Prediction Workflow



---

# Overfitting Problem

ANNs may memorize training data.

This causes:
- poor generalization.

---

# Regularization Techniques

Used to reduce overfitting:
- Dropout
- Early Stopping
- L2 Regularization

---

# Dropout Layer

Dropout randomly disables neurons during training.

Benefits:
- improves generalization,
- prevents overfitting.

---

# Industry Perspective

Customer churn prediction systems are part of:
- business intelligence,
- recommendation systems,
- customer analytics pipelines.

Modern companies use:
- Deep Learning,
- Big Data,
- Real-Time AI systems

for predictive analytics.

---

# TensorFlow/Keras Implementation

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Build ANN
model = Sequential()

# Hidden Layers
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))

# Output Layer
model.add(Dense(1, activation='sigmoid'))

# Compile Model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print(model.summary())
```

---

# Why Adam Optimizer?

Adam combines:
- Momentum,
- Adaptive Learning Rates.

Advantages:
- faster convergence,
- stable optimization.

Industry standard for many DL tasks.

---

# Why ANN Beats Traditional ML Sometimes?

ANNs:
- learn non-linear relationships,
- automatically discover hidden patterns,
- scale with large datasets.

Traditional ML often requires:
- manual feature engineering.

---

# Limitations of ANN

- Requires large datasets
- Computationally expensive
- Risk of overfitting
- Harder interpretability

---

# Real World Challenges

Industry systems face:
- imbalanced datasets,
- noisy customer data,
- real-time inference requirements,
- fairness concerns,
- privacy regulations.

---

# Important Terms

| Term | Meaning |
|---|---|
| Churn | Customer leaving service |
| ANN | Artificial Neural Network |
| Epoch | One complete training cycle |
| BCE Loss | Binary classification loss |
| Dropout | Regularization technique |
| Feature Scaling | Normalizing data |

---

# Interview Questions

## Q1. Why is sigmoid used in churn prediction?
Because churn prediction is binary classification.

## Q2. Why use BCE loss?
Because it works well with probability outputs.

## Q3. Why is feature scaling important?
Neural networks train more efficiently on scaled data.

## Q4. What is overfitting?
When a model memorizes training data instead of generalizing.

---

# Key Points Learned Today

✅ Learned customer churn prediction  
✅ Understood ANN architecture  
✅ Learned preprocessing pipeline  
✅ Understood BCE loss and sigmoid  
✅ Learned evaluation metrics  
✅ Understood overfitting and dropout  
✅ Connected Deep Learning with business analytics

---

# Mini Summary

Customer Churn Prediction using ANN is a powerful real-world Deep Learning application where neural networks learn hidden customer behavior patterns to predict whether users may leave a service. Modern businesses rely heavily on such AI systems for retention, analytics, and revenue optimization.

---

# Progress

✅ Completed Day 10 - Customer Churn Prediction Using Artificial Neural Network (ANN)
