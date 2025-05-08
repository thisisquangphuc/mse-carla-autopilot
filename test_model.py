import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2

# Load the model
model = tf.keras.models.load_model('models/pedestrian_model.keras')

IMG_SIZE = 224
THRESHOLD = 0.5

# Load and preprocess test image
image_path = 'test_resources/001_pedes.png'  # Replace with your test image
image = cv2.imread(image_path)
if image is None:
    raise ValueError(f"Image not found: {image_path}")

resized_image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
normalized_image = resized_image / 255.0
input_array = np.expand_dims(normalized_image, axis=0)

# Predict
prediction = model.predict(input_array)[0][0]

# Decision
if prediction > THRESHOLD:
    message = "Pedestrian found."
else:
    message = "No pedestrian detected."

# Display result
cv2.putText(image, message, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if prediction < THRESHOLD else (0, 0, 255), 2)
cv2.imshow("Result", image)
cv2.waitKey(0)
cv2.destroyAllWindows()