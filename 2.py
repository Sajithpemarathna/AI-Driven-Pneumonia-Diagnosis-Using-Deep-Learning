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

    # Step 1: Data Augmentation to prevent overfitting
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Step 2: Define the CNN model
    model = Sequential()

    # Add Convolutional Layers and MaxPooling Layers
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

    # Step 3: Train the model
    history = model.fit(
        datagen.flow(train_images, train_labels, batch_size=16),
        epochs=20,
        validation_data=(val_images, val_labels),
        callbacks=[early_stopping],
        verbose=1
    )

    # Save the trained model for future use
    model.save(model_path)
