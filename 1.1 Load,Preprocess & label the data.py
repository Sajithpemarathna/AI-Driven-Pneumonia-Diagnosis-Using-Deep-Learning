import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Preprocessing Function to convert to Grayscale, Resize, & Normalize images
def preprocess_image(image_path, img_size=224):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read the image and convert into Grayscale
    image = cv2.resize(image, (img_size, img_size))       # Resize the image into uniform size 224x224
    image = image / 255.0  # Normalize pixel values to [0, 1] range
    return image

# Function for images loading and labeling from a folder
def load_data_from_folder(folder_path, img_size=224):
    images = []
    labels = []
    categories = ['PNEUMONIA', 'NORMAL'] # Defining 'PNEUMONIA' & 'NORMAL' Categories 
    
    for category in categories:
        category_path = os.path.join(folder_path, category)
        label = 1 if category == 'PNEUMONIA' else 0  # Assigning 'PNEUMONIA' patients as 1 & 'NORMAL' patients 0
        for file_name in os.listdir(category_path):
            file_path = os.path.join(category_path, file_name)  
            image = preprocess_image(file_path, img_size) # Preprocess images such as Grascale, Resize
            images.append(image)
            labels.append(label)
    
    return np.array(images), np.array(labels) # return the converted images & lables into Numpy Arrays

# Load data to train, validate, and test
dataset_dir = r"C:\Users\PC\OneDrive\Desktop\MSc Data Analytics\Computer Vision and Artificial Intelligence\8 Assignments\chest_xray" #Dataset path in local desktop
train_images, train_labels = load_data_from_folder(os.path.join(dataset_dir, 'train'))
val_images, val_labels = load_data_from_folder(os.path.join(dataset_dir, 'val'))
test_images, test_labels = load_data_from_folder(os.path.join(dataset_dir, 'test'))

# Reshape the dataset for Neural Network Input
train_images = train_images.reshape(-1, 224, 224, 1)
val_images = val_images.reshape(-1, 224, 224, 1)
test_images = test_images.reshape(-1, 224, 224, 1)



# Function to display preprocessed  10 images
def display_sample_images(images, labels, sample_size=10):
    plt.figure(figsize=(12, 6)) # Create a new figure with a size of 12*6 inches
    for i in range(sample_size):
        plt.subplot(2, 5, i + 1) # Creating Subplots in 2 rows
        plt.imshow(images[i].reshape(224, 224), cmap='gray')
        plt.title('PNEUMONIA' if labels[i] == 1 else 'NORMAL') # Subplots labeling
        plt.axis('off') # Turn off the axis for better visualization
    plt.show()

# Display images from the test set
#display_sample_images(test_images, test_labels)

