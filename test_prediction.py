import cv2
import numpy as np
import tensorflow as tf
import os

# Load model
model = tf.keras.models.load_model("model/fruit_model.h5")

# Load class labels
train_dir = "dataset/fruits-360/Training"
class_labels = sorted(os.listdir(train_dir))

# Image input size
img_height, img_width = 100, 100

# Start webcam
cap = cv2.VideoCapture(0)
print("Press 's' to snap and predict | Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Live Feed", frame)

    key = cv2.waitKey(1)

    if key == ord('s'):
        # Resize and preprocess
        resized = cv2.resize(frame, (img_width, img_height))
        normalized = resized / 255.0
        reshaped = np.reshape(normalized, (1, img_height, img_width, 3))

        # Predict
        prediction = model.predict(reshaped)
        predicted_index = np.argmax(prediction)
        predicted_label = class_labels[predicted_index]

        print(f"Predicted Fruit: {predicted_label}")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
