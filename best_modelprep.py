try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    import os
except ModuleNotFoundError:
    print("Error: TensorFlow is not installed. Please install it using 'pip install tensorflow'.")
    raise SystemExit("TensorFlow not installed")

# Dataset path
dataset_path = "C:\\Users\\bobyh\\Downloads\\Medical-Image-Classification-using-CNN-main\\Medical-Image-Classification-using-CNN-main\\chest_xray"

# Image preprocessing
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = datagen.flow_from_directory(
    os.path.join(dataset_path, "train"),
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    os.path.join(dataset_path, "train"),
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

# Define the model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_generator, epochs=10, validation_data=validation_generator)

# Save the model
model.save("best_model.h5")

# Evaluate the model
loss, accuracy = model.evaluate(validation_generator)
print(f"Model training complete. Validation Accuracy: {accuracy:.2f}, Loss: {loss:.2f}")
