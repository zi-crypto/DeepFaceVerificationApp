# Deep Facial Verification App

## Overview

This repository contains the code and resources for a **Deep Facial Verification App** that utilizes a Siamese Neural Network (SNN) architecture. The app is designed to determine whether two facial images belong to the same individual by learning and comparing feature embeddings through a similarity metric. The project also incorporates fine-tuning techniques to enhance performance.

## Achievements

- **1st Place in Academic Projects:** This project was submitted as part of an academic course and was awarded 1st place among the best projects.
- You can see our Submission Here: https://drive.google.com/drive/folders/1mMOt5V0qF6Gr5YFqf2cjMQvmd9Vicjbp

## Features

- **Siamese Neural Network Architecture:** Employs a twin neural network structure that shares weights to learn and compare embeddings of facial images. :contentReference[oaicite:0]{index=0}
- **Advanced Fine-Tuning:** Implements techniques such as learning rate scheduling, data augmentation, and loss function optimization to improve model robustness and accuracy.
- **High Accuracy:** Achieves reliable facial verification even with challenging datasets.
- **User-Friendly Interface:** Provides a clean and intuitive interface for easy use and testing.

## How It Works

1. **Input:** The user provides two facial images.
2. **Embedding Generation:** Each image is processed through the twin networks to generate feature embeddings.
3. **Similarity Computation:** A distance metric (e.g., Euclidean or cosine) calculates the similarity between embeddings.
4. **Verification Decision:** The app determines whether the two images belong to the same individual based on a predefined similarity threshold.

## Installation

### Prerequisites

- Python 3.8+
- TensorFlow
- OpenCV
- NumPy

## Dataset
The project was developed using a curated facial dataset based on The Famous LFW Dataset (Labeled Faces in the Wild Dataset ) with balanced positive and negative pairs. Preprocessing steps include face detection, alignment, and normalization to ensure consistent input to the model.

## Model Architecture
 . **Base Model:** A Convolutional Neural Network General Architecture ( Not Pretrained )
 . **Feature Extractor:** Subsequent layers that distill the base model's output into a compact and informative embedding vector.
 . **Output Layer:** Computes the similarity score between embeddings using L1 distance metric, facilitating the verification process.

## Fine-Tuning Techniques
1. **Data Augmentation:** Applies transformations such as rotation, scaling, and flipping to the training data to improve model generalization.
2. **Newly Added Data with Different Lighting Systems**

## Evaluation Metrics
 . **Precision**
 . **Recall**

## Results
 . Achieved state-of-the-art results on the test set with high precision and recall.
 . Demonstrated robustness to variations in lighting, pose, and facial expressions.

## ADDED Efficency
 . Worked in Minimizing the Verification time from almost 18 seconds to 7 seconds by batching the verification images and verifying them all in the same time ( Without using GPU )

Thank you for exploring this project! 🚀
