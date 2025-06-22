# AI-Driven-Pneumonia-Diagnosis-Using-Deep-Learning
# AI-Driven Pneumonia Diagnosis: Chest X-ray Classification

## 🧠 Project Overview
This project focuses on developing an AI model for automated detection of pneumonia from chest X-ray images using deep learning. Built as part of the MSc in Data Analytics (Computer Vision & AI module), this solution highlights the real-world application of convolutional neural networks in medical imaging.

## 💡 Key Features
- Implemented both a custom CNN and ResNet50 transfer learning model using TensorFlow and Keras.
- Preprocessed data using normalization and real-time augmentation with `ImageDataGenerator`.
- Used callbacks like **EarlyStopping** and **ReduceLROnPlateau** to improve training efficiency.
- Evaluated model performance with accuracy, precision, recall, F1-score, and AUC-ROC metrics.
- Final model achieved **~89% accuracy** with excellent generalization on validation data.

## 📦 Dataset Used
- **Source**: [Kaggle - Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- Includes labeled chest X-ray images: Normal vs Pneumonia.

## 🛠️ Tools & Technologies
Python · TensorFlow · Keras · Google Colab · Scikit-learn · Matplotlib · ResNet50 · CNN · ImageDataGenerator

## 📊 Model Performance
- Accuracy: ~89%
- AUC-ROC: 0.87
- F1 Score: High performance across all classes
- Visualized results using ROC curves and confusion matrix.


