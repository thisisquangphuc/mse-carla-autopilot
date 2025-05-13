#### Import libs
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

print(tf.__version__)

import os
import shutil
from sklearn.model_selection import train_test_split

base_dir = 'dataset/'
train_dir = 'dataset/train'
val_dir = 'dataset/validation'

# Clear previous splits (if any)
if os.path.exists(train_dir):
    shutil.rmtree(train_dir)
if os.path.exists(val_dir):
    shutil.rmtree(val_dir)

os.makedirs(train_dir + '/pedestrian')
os.makedirs(train_dir + '/no_pedestrian')
os.makedirs(val_dir + '/pedestrian')
os.makedirs(val_dir + '/no_pedestrian')

# Helper function to split images
def split_data(src_dir, train_dst, val_dst, split_ratio=0.8):  # <==== 80-20 HERE
    images = os.listdir(src_dir)
    train_imgs, val_imgs = train_test_split(images, train_size=split_ratio, random_state=42)

    for img in train_imgs:
        shutil.copy(os.path.join(src_dir, img), os.path.join(train_dst, img))
    for img in val_imgs:
        shutil.copy(os.path.join(src_dir, img), os.path.join(val_dst, img))

split_data(base_dir + '/pedestrian', train_dir + '/pedestrian', val_dir + '/pedestrian')
split_data(base_dir + '/no_pedestrian', train_dir + '/no_pedestrian', val_dir + '/no_pedestrian')


# Set the image size (ResNet50 expects at least 197x197, but we usually use 224x224)
IMG_SIZE = 224
BATCH_SIZE = 32

# Data generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    zoom_range=0.2,
    horizontal_flip=True
)

# train_datagen = ImageDataGenerator(
#     rescale=1./255,
#     rotation_range=20,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     shear_range=0.1,
#     zoom_range=0.2,
#     horizontal_flip=True,
#     brightness_range=[0.7, 1.3],
#     fill_mode='nearest'
# )

val_datagen = ImageDataGenerator(rescale=1./255)

# Load images from directories
train_generator = train_datagen.flow_from_directory(
    directory=train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

val_generator = val_datagen.flow_from_directory(
    directory=val_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)


##### Resnet50
base_model = tf.keras.applications.ResNet50(
    include_top=False,
    weights='imagenet',
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Binary: pedestrian vs. not
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

# Optional: add early stopping to prevent overfitting
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=2,
    restore_best_weights=True
)

# Train the model
# history = model.fit(train_generator, validation_data=val_generator, epochs=5)
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=20,                     # feel free to increase for better results
    callbacks=[early_stop]
)

plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy over epochs')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss over epochs')
plt.legend()
plt.show()

model.save('model/pedestrian_model.keras')
new_model=tf.keras.models.load_model('model/pedestrian_model.keras')