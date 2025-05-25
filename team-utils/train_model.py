import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
import os
import shutil
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping

# Set constants
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10


base_dir = 'dataset/traffic_sign'
train_dir = 'dataset/local/train'
val_dir = 'dataset/local/val'

# # Clear previous splits (if any)
# if os.path.exists(train_dir):
#     shutil.rmtree(train_dir)
# if os.path.exists(val_dir):
#     shutil.rmtree(val_dir)

# # Helper function to split images
# def split_data(src_dir, train_dst, val_dst, split_ratio=0.8):  # <==== 80-20 HERE
#     images = os.listdir(src_dir)
#     train_imgs, val_imgs = train_test_split(images, train_size=split_ratio, random_state=42)

#     for img in train_imgs:
#         shutil.copy(os.path.join(src_dir, img), os.path.join(train_dst, img))
#     for img in val_imgs:
#         shutil.copy(os.path.join(src_dir, img), os.path.join(val_dst, img))

# def split_dataset_classes(base_path, train_path, val_path, split_ratio=0.8):
#     class_names = [cls for cls in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, cls))]
#     for cls in class_names:
#         src = os.path.join(base_path, cls)
#         dst_train = os.path.join(train_path, cls)
#         dst_val = os.path.join(val_path, cls)
#         os.makedirs(dst_train, exist_ok=True)
#         os.makedirs(dst_val, exist_ok=True)
#         split_data(src, dst_train, dst_val, split_ratio)

# split_dataset_classes(base_dir, train_dir, val_dir, split_ratio=0.8)

# Preprocessing

# Data generators
# train_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)


train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_gen = val_datagen.flow_from_directory(
    val_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Build the model
base_model = ResNet50(include_top=False, weights='imagenet', input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_model.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dense(train_gen.num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

# Train
history = model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS)

# Save the model to models/
# model.save('models/resnet50_pedestrian_tf2-11_py3-7.keras', save_format='keras')
model.save('models/resnet50_sign.keras', save_format='keras')