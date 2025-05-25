import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
import os
# Load the model
# model = tf.keras.models.load_model('models/resnet50_pedestrian_tf2-11_py3-7-v1.h5')
model = tf.keras.models.load_model('models/resnet50_sign.keras')

class_names = ['speed_30', 'speed_60', 'speed_90', 'speed_limit_30', 'speed_limit_40', 'speed_limit_60', 'stop', 'back']

IMG_SIZE = 224
THRESHOLD = 0.5

# Load and preprocess test image
test_dir = 'test_resources/sign/'
# image_path = 'test_resources/001_nopedes.png'

#read test images from directory
for fname in os.listdir(test_dir):
    if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue
    image_path = os.path.join(test_dir, fname)
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found: {image_path}")

    resized_image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    normalized_image = resized_image / 255.0
    input_array = np.expand_dims(normalized_image, axis=0)

    # Predict
    prediction_scores = model.predict(input_array)[0]
    predicted_class_index = np.argmax(prediction_scores)
    print(f"{fname} — Prediction scores: {prediction_scores}")
    # Decision
    message = f"Predicted: {class_names[predicted_class_index]}"

    # Display result
    # cv2.putText(image, message, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if prediction < THRESHOLD else (0, 0, 255), 2)
    # cv2.imshow("Result", image)

    # Resize original image for display
    display_image = cv2.resize(image, (360, 360))
    model_input = cv2.resize(image, (360, 360))  # same size, not normalized, just for visual

    # cv2.putText(display_image, message, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if prediction < THRESHOLD else (0, 0, 255), 2)
    # cv2.imshow("Result", display_image)

    # Annotate prediction on display image
    cv2.putText(display_image, message, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 255, 0), 2)

    # Combine both images side by side
    combined = np.hstack((model_input, display_image))

    # Show the combined image
    cv2.imshow("Input (left) vs Prediction (right)", combined)


    cv2.waitKey(0)
    cv2.destroyAllWindows()