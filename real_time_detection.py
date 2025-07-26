import cv2
import numpy as np
import tensorflow as tf
import os

# Load model and class labels
model = tf.keras.models.load_model("model/fruit_model.h5")
train_dir = "dataset/fruits-360/Training"
class_labels = sorted(os.listdir(train_dir))

# Image input size
img_height, img_width = 100, 100

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape

    # Define center square (region of interest)
    box_size = 200
    x1 = w // 2 - box_size // 2
    y1 = h // 2 - box_size // 2
    x2 = x1 + box_size
    y2 = y1 + box_size

    # Draw rectangle
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Crop region inside rectangle
    roi = frame[y1:y2, x1:x2]
    resized = cv2.resize(roi, (img_width, img_height))
    normalized = resized / 255.0
    reshaped = np.reshape(normalized, (1, img_height, img_width, 3))

    # Predict
    prediction = model.predict(reshaped)
    predicted_index = np.argmax(prediction)
    predicted_label = class_labels[predicted_index]

    # Overlay predicted label
    cv2.putText(frame, f"Detected: {predicted_label}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Real-Time Fruit Detection", frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
