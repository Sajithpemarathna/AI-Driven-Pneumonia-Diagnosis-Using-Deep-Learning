import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Sequential

# Step 1: Initialize the model
model = Sequential()

# Step 2: Add the first Convolutional Layer
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 1)))
# Explanation: 
# - Conv2D: A convolutional layer that learns 32 filters (features) using 3x3 kernels.
# - ReLU activation function introduces non-linearity.
# - Input shape is (224, 224, 1) to match the grayscale images of 224x224 pixels.

# Step 3: Add a MaxPooling Layer
model.add(MaxPooling2D(pool_size=(2, 2)))
# Explanation:
# - MaxPooling: Reduces the spatial dimensions of the feature maps (224x224 becomes 112x112) 
#   while keeping the most important information. This makes the model more efficient.

# Step 4: Add the second Convolutional Layer
model.add(Conv2D(64, (3, 3), activation='relu'))
# Explanation:
# - Conv2D with 64 filters, which allows the model to learn more complex patterns in the images.

# Step 5: Add a second MaxPooling Layer
model.add(MaxPooling2D(pool_size=(2, 2)))
# Explanation: 
# - MaxPooling again reduces the size of the feature maps (112x112 becomes 56x56), 
#   keeping the most important features and reducing computational cost.

# Step 6: Add the third Convolutional Layer
model.add(Conv2D(128, (3, 3), activation='relu'))
# Explanation:
# - Conv2D with 128 filters to learn more complex and abstract features like texture or shape anomalies that might be present in pneumonia cases.

# Step 7: Add a third MaxPooling Layer
model.add(MaxPooling2D(pool_size=(2, 2)))
# Explanation:
# - MaxPooling again reduces the size of the feature maps (56x56 becomes 28x28).

# Step 8: Flatten the feature maps
model.add(Flatten())
# Explanation:
# - Flattening converts the 3D feature maps into a 1D vector. This is necessary to pass the data into fully connected (Dense) layers.

# Step 9: Add a fully connected layer (Dense Layer)
model.add(Dense(128, activation='relu'))
# Explanation:
# - Dense layer with 128 units. This layer helps the model combine the features learned by the convolutional layers and make a final decision about the presence of pneumonia.

# Step 10: Add the Output Layer
model.add(Dense(1, activation='sigmoid'))
# Explanation:
# - Sigmoid is used because this is a binary classification task. It outputs a probability between 0 and 1 (1 being pneumonia and 0 being normal).

# Step 11: Compile the Model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# Explanation:
# - Adam optimizer is used because it adapts the learning rate during training, making it efficient for training deep neural networks.
# - Binary cross-entropy is the loss function, appropriate for binary classification tasks.
# - We track accuracy as a performance metric to monitor the model's success.

# Step 12: Model Summary
model.summary()
# Explanation:
# - The model summary displays the architecture, showing the number of parameters and the structure of each layer.

# Load the dataset (you should already have this prepared)
# Assuming train_images, train_labels, val_images, val_labels are already loaded

# Step 13: Train the model
history = model.fit(train_images, train_labels, epochs=10, validation_data=(val_images, val_labels), batch_size=32)
# Explanation:
# - We train the model for 10 epochs, meaning the model will pass over the training data 10 times.
# - Validation data is provided to evaluate the model on unseen data after each epoch to prevent overfitting.
# - Batch size of 32 means the model updates its parameters after seeing 32 samples at a time.

# Step 14: Evaluate the Model on Test Data
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'Test accuracy: {test_acc}')
# Explanation:
# - After training, we evaluate the model on the test data to see how well it generalizes to new, unseen examples.
