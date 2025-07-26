import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout


# Set the paths to your dataset folders
train_dir = "dataset/fruits-360/Training"
test_dir = "dataset/fruits-360/Test"

# Batch train
img_height, img_width = 100, 100
batch_size = 32

# Normalizing
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# generator for training images
# flow_from_directory(): read images from a structured directory
train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width), # resize each img
    batch_size=batch_size,
    class_mode='categorical' # labels will be one-hot encoded
)

test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

# Define the CNN model
model = Sequential()

# Hidden Layers
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Output layer
model.add(Flatten()) # Converts to 1D vector
model.add(Dense(128, activation='relu')) # Fully Connected dense layer
model.add(Dropout(0.3))
model.add(Dense(train_data.num_classes, activation='softmax'))

# Training
# Compile the model
model.compile(
    optimizer='adam', # Adam (Adaptive Moment Estimation)
    loss='categorical_crossentropy', # how "wrong" your model's predictions are compared to the actual correct labels
    metrics=['accuracy'] # Evaluation
)

# Train the model
epochs = 10

history = model.fit(
    train_data,
    epochs=epochs, # number of complete passes through the entire training dataset
    validation_data=test_data
)

# Create the model directory if it doesn't exist
os.makedirs("model", exist_ok=True)

# Save the trained model
model.save("model/fruit_model.h5")
