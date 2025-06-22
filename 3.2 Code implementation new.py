import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Define directories for training, validation, and testing datasets
train_dir = r"C:\Users\PC\OneDrive\Desktop\MSc Data Analytics\Computer Vision and Artificial Intelligence\8 Assignments\chest_xray\train"  # Update with the correct path
test_dir = r"C:\Users\PC\OneDrive\Desktop\MSc Data Analytics\Computer Vision and Artificial Intelligence\8 Assignments\chest_xray\test"    # Update with the correct path
val_dir = r"C:\Users\PC\OneDrive\Desktop\MSc Data Analytics\Computer Vision and Artificial Intelligence\8 Assignments\chest_xray\val"  # Update with the correct path

# Define categories (classes) in the dataset
categories = ['NORMAL', 'PNEUMONIA']

# Set the image size to match the input shape in the CNN model
IMG_SIZE = 128

# Function to load images, resize them, and assign labels (0 for NORMAL, 1 for PNEUMONIA)
def load_data(data_dir):
    data = []
    for category in categories:
        path = os.path.join(data_dir, category)
        class_num = categories.index(category)  # 0 for NORMAL, 1 for PNEUMONIA
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)  # Load image as grayscale
                resized_img = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # Resize image to 128x128
                data.append([resized_img, class_num])
            except Exception as e:
                pass
    return data

# Load and preprocess the training, validation, and test datasets
train_data = load_data(train_dir)
test_data = load_data(test_dir)
val_data = load_data(val_dir)

# Shuffle the data to mix the NORMAL and PNEUMONIA examples
import random
random.shuffle(train_data)
random.shuffle(test_data)
random.shuffle(val_data)

# Separate the images and labels for training, validation, and test datasets
train_images = np.array([i[0] for i in train_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
train_labels = np.array([i[1] for i in train_data])

test_images = np.array([i[0] for i in test_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
test_labels = np.array([i[1] for i in test_data])

val_images = np.array([i[0] for i in val_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
val_labels = np.array([i[1] for i in val_data])

# Normalize the pixel values to the range 0-1 by dividing by 255
train_images = train_images / 255.0
test_images = test_images / 255.0
val_images = val_images / 255.0

# Plot a few images to verify the data loading process (optional)
def plot_sample_images(images, labels, category_names, sample_size=14):
    plt.figure(figsize=(10, 6))
    for i in range(sample_size):
        plt.subplot(2, 7, i+1)
        plt.imshow(images[i].reshape(IMG_SIZE, IMG_SIZE), cmap='gray')
        plt.title(category_names[labels[i]])
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Plot sample images from the training dataset
plot_sample_images(train_images, train_labels, categories)


import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

# Step 1: Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=20,        # Rotate images by 20 degrees
    width_shift_range=0.2,    # Shift images horizontally by 20%
    height_shift_range=0.2,   # Shift images vertically by 20%
    shear_range=0.2,          # Shear transformations
    zoom_range=0.2,           # Zoom images by 20%
    horizontal_flip=True,     # Flip images horizontally
    fill_mode='nearest'       # Fill in missing pixels after transformations
)

# Step 2: Initialize the model
model = Sequential()

# Step 3: Add the first Convolutional Layer with reduced image size (128x128) and fewer filters
model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(128, 128, 1)))  # Reduced filters to 16

# Step 4: Add MaxPooling Layer
model.add(MaxPooling2D(pool_size=(2, 2)))

# Step 5: Add the second Convolutional Layer with fewer filters
model.add(Conv2D(32, (3, 3), activation='relu'))  # Reduced filters to 32

# Step 6: Add MaxPooling Layer
model.add(MaxPooling2D(pool_size=(2, 2)))

# Step 7: Add the third Convolutional Layer with fewer filters
model.add(Conv2D(64, (3, 3), activation='relu'))  # Reduced filters to 64

# Step 8: Add MaxPooling Layer
model.add(MaxPooling2D(pool_size=(2, 2)))

# Step 9: Flatten the feature maps
model.add(Flatten())  # The output shape before Flattening is (16, 16, 64), so Flatten results in 16384 units.

# Step 10: Add Fully Connected Layer with Dropout
model.add(Dense(128, activation='relu'))  # Fully connected layer with 128 neurons
model.add(Dropout(0.5))  # Dropout layer to prevent overfitting

# Step 11: Output Layer
model.add(Dense(1, activation='sigmoid'))  # Sigmoid for binary classification

# Step 12: Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Step 13: Add Early Stopping to avoid overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Step 14: Train the model using augmented data with smaller batch size (16) and fewer epochs (20)
history = model.fit(
    datagen.flow(train_images, train_labels, batch_size=16),  # Smaller batch size
    epochs=20,                   # Reduced number of epochs to 20
    validation_data=(val_images, val_labels),
    callbacks=[early_stopping],   # Early stopping to prevent overfitting
    verbose=1
)

# Step 15: Evaluate the Model on Test Data
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'Test accuracy: {test_acc}')
