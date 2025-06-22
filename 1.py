import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import random

# Step 1: Load and Preprocess the Data

# Define directories for training, validation, and testing datasets
train_dir = r"C:\Users\PC\OneDrive\Desktop\MSc Data Analytics\Computer Vision and Artificial Intelligence\8 Assignments\chest_xray\train"
test_dir = r"C:\Users\PC\OneDrive\Desktop\MSc Data Analytics\Computer Vision and Artificial Intelligence\8 Assignments\chest_xray\test"
val_dir = r"C:\Users\PC\OneDrive\Desktop\MSc Data Analytics\Computer Vision and Artificial Intelligence\8 Assignments\chest_xray\val"

# Define categories (classes) in the dataset
categories = ['NORMAL', 'PNEUMONIA']

# Set the image size to 128x128 to match the CNN input shape
IMG_SIZE = 128

# Load and preprocess the images
def load_data(data_dir):
    data = []
    for category in categories:
        path = os.path.join(data_dir, category)
        class_num = categories.index(category)  # 0 for NORMAL, 1 for PNEUMONIA
        print(f"Loading {category} images from {data_dir}...")
        for img in os.listdir(path):
            try:
                img_path = os.path.join(path, img)
                if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):  # Ensure the file is an image
                    img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img_array is not None:
                        resized_img = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # Resize image to 128x128
                        data.append([resized_img, class_num])
            except Exception as e:
                print(f"Error loading image {img}: {e}")
    return data

# Load the data
train_data = load_data(train_dir)
test_data = load_data(test_dir)
val_data = load_data(val_dir)

# Shuffle the data
random.shuffle(train_data)
random.shuffle(test_data)
random.shuffle(val_data)

# Separate images and labels
train_images = np.array([i[0] for i in train_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
train_labels = np.array([i[1] for i in train_data])

test_images = np.array([i[0] for i in test_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
test_labels = np.array([i[1] for i in test_data])

val_images = np.array([i[0] for i in val_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
val_labels = np.array([i[1] for i in val_data])

# Normalize pixel values to the range [0, 1]
train_images = train_images / 255.0
test_images = test_images / 255.0
val_images = val_images / 255.0

# Step 2: Define, Train, and Save the Model

# Define the model path where the trained model will be saved
model_path = 'pneumonia_cnn_model.h5'

# Check if the model file already exists
if os.path.exists(model_path):
    # If the model is already saved, load it
    print("Loading saved model...")
    model = load_model(model_path)
else:
    # If the model file doesn't exist, define and train the model
    print("Training model and saving it for future use...")

    # Data Augmentation to prevent overfitting
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Define the CNN model
    model = Sequential()
    model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(128, 128, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Flatten the feature maps to feed into the fully connected layers
    model.add(Flatten())

    # Add a fully connected layer and dropout for regularization
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))

    # Output layer for binary classification
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Add Early Stopping to avoid overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Train the model
    history = model.fit(
        datagen.flow(train_images, train_labels, batch_size=16),
        epochs=20,
        validation_data=(val_images, val_labels),
        callbacks=[early_stopping],
        verbose=1
    )

    # Save the trained model for future use
    model.save(model_path)

# Step 3: Evaluate the model on the test data
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'Test accuracy: {test_acc}')
