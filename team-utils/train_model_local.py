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
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.callbacks import ModelCheckpoint
import time

# Set constants
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 15
VERSION = time.strftime("%Y%m%d-%H%M%S")

base_dir = 'dataset/'
train_dir = 'dataset/train'
val_dir = 'dataset/val'
output_dir = 'local_output/'

def check_corrupted_images(base_path):
    for root, dirs, files in os.walk(base_path):
        for fname in files:
            file_path = os.path.join(root, fname)
            try:
                with Image.open(file_path) as img:
                    img.verify()  # Only verifies; doesn't decode
            except Exception as e:
                print(f"[CORRUPT] {file_path} — {e}")
                # remove
                os.remove(file_path)
                print(f"[REMOVED] {file_path}")

check_corrupted_images(base_dir)            

# Clear previous splits (if any)
if os.path.exists(train_dir):
    shutil.rmtree(train_dir)
if os.path.exists(val_dir):
    shutil.rmtree(val_dir)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
# Helper function to split images
def split_data(src_dir, train_dst, val_dst, split_ratio=0.8):  # <==== 80-20 HERE
    images = os.listdir(src_dir)
    train_imgs, val_imgs = train_test_split(images, train_size=split_ratio, random_state=42)

    for img in train_imgs:
        shutil.copy(os.path.join(src_dir, img), os.path.join(train_dst, img))
    for img in val_imgs:
        shutil.copy(os.path.join(src_dir, img), os.path.join(val_dst, img))

def split_dataset_classes(base_path, train_path, val_path, split_ratio=0.8):
    class_names = [cls for cls in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, cls))]
    for cls in class_names:
        src = os.path.join(base_path, cls)
        dst_train = os.path.join(train_path, cls)
        dst_val = os.path.join(val_path, cls)
        os.makedirs(dst_train, exist_ok=True)
        os.makedirs(dst_val, exist_ok=True)
        split_data(src, dst_train, dst_val, split_ratio)

split_dataset_classes(base_dir, train_dir, val_dir, split_ratio=0.8)

# Preprocessing

# Data generators
# train_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    zoom_range=0.2,
    horizontal_flip=True
)
# train_datagen = ImageDataGenerator(
#     rescale=1./255,
#     rotation_range=25,
#     zoom_range=0.3,
#     brightness_range=[0.8, 1.2],
#     horizontal_flip=True
# )

val_datagen = ImageDataGenerator(rescale=1./255)


train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=True
)
# Debug
print(train_gen.class_indices)

val_gen = val_datagen.flow_from_directory(
    val_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# Build the model
base_model = ResNet50(include_top=False, weights='imagenet', input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Binary: pedestrian vs. not
])

################# Initial Training #################
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

model_path = os.path.join(output_dir, 'trained_models/')
#  check if the model path exists, if not, create it
if not os.path.exists(model_path):
    os.makedirs(model_path)
    
output_model = os.path.join(model_path, VERSION + 'resnet50_pedestrian_before_finetune.keras')
checkpoint = ModelCheckpoint(output_model, monitor='val_loss', save_best_only=True)

# Enable early stopping - to prevent overfitting
early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

# Add to callbacks:
callbacks=[early_stop, checkpoint]

## Train - with early stopping
history = model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS, callbacks=callbacks)
## Train - without early stopping
# history = model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS)
    
################ Fine-tuning (Optional)#################

base_model.trainable = True  # Unfreeze base model

# Freeze all layers except the last 20 of ResNet
for layer in base_model.layers[:-20]:
    layer.trainable = False

# Re-compile with a lower learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Fine-tune for a few additional epochs
fine_tune_history = model.fit(train_gen, validation_data=val_gen, epochs=5, callbacks=callbacks)

################# Evaluation #################
# Evaluate
loss, accuracy = model.evaluate(val_gen)
print(f'Validation accuracy: {accuracy:.2f}')

plot_path = os.path.join(output_dir, 'plots/')
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

# Plot
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy over epochs')
plt.legend()
# plt.show()
plt.savefig(plot_path + VERSION + 'accuracy.png')


plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss over epochs')
plt.legend()
# plt.show()
plt.savefig(plot_path + VERSION + 'loss.png')

# Combine histories
def combine_history(h1, h2):
    combined = {}
    for k in h1.history:
        combined[k] = h1.history[k] + h2.history.get(k, [])
    return combined

combined_history = combine_history(history, fine_tune_history)

# Then plot using combined_history
plt.figure()
plt.plot(combined_history['accuracy'], label='Train Acc')
plt.plot(combined_history['val_accuracy'], label='Val Acc')
plt.title('Combined Accuracy')
plt.legend()
plt.savefig(plot_path + VERSION + 'combined_accuracy.png')

plt.figure()
plt.plot(combined_history['loss'], label='Train Loss')
plt.plot(combined_history['val_loss'], label='Val Loss')
plt.title('Combined Loss')
plt.legend()
plt.savefig(plot_path + VERSION + 'combined_loss.png')

################# Save the model #################
# Save the model to models/
output_model = os.path.join(model_path, VERSION + 'renet50_pedestrian_after_finetune.keras')
model.save(output_model)
# model.save('local/trained_models/resnet50_sign.keras', save_format='keras')
