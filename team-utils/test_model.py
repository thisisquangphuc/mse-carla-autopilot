import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, f1_score

# Load the model
# model = tf.keras.models.load_model('local/trained_models/resnet50_pedestrian_20250527-082525.keras')
model = tf.keras.models.load_model('best_model.keras')
# model = tf.keras.models.load_model('models/resnet50_sign.keras')

class_names = ['pedestrian', 'no_pedestrian']

IMG_SIZE = 224
THRESHOLD = 0.5

# Load and preprocess test image
test_dir = 'test_resources/'
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
    if prediction_scores[predicted_class_index] < THRESHOLD:
        message = f"Image (no pedestrian)"
    else:
        message = f"Image (pedestrian)"
        
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