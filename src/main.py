import io
import json
import os
import sys

import wandb
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint

data_dir = "./dataset"
labels = os.listdir(data_dir)

images = []
for label in labels:
    label_dir = os.path.join(data_dir, label)
    for img_file in os.listdir(label_dir):
        img_path = os.path.join(label_dir, img_file)
        img = Image.open(img_path)
        img_copy = img.copy()
        img.close()
        img_copy = img_copy.resize(size=(400, 400))
        images.append((img_copy, label))

print("Finished loading images to PIL objects.")

X = []
y = []

for image, label in images:
    if np.array(image).shape == (400, 400, 3):
        X.append(np.array(image))
        y.append(label)

X = np.array(X)
y = np.array(y)

le = LabelEncoder()
y_encoded = le.fit_transform(y)

y_encoded = tf.keras.utils.to_categorical(y_encoded, num_classes=5)

label_dict = dict(zip(le.classes_, le.transform(le.classes_)))
print(label_dict)

raw_X_train_val, raw_X_test, raw_y_train_val, raw_y_test = train_test_split(X, y_encoded, test_size=0.1,
                                                                            random_state=42,
                                                                            stratify=y_encoded)
raw_X_train, raw_X_val, raw_y_train, raw_y_val = train_test_split(raw_X_train_val, raw_y_train_val, test_size=0.2,
                                                                  random_state=42,
                                                                  stratify=raw_y_train_val)

wandb.init(project="SZUM", name="SPLIT1-ES")

print("started_training..")

# Define the model architecture
model = tf.keras.Sequential([
    tf.keras.applications.MobileNetV3Large(input_shape=(224, 224, 3), include_top=False,
                                           weights='imagenet'),
    # tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax')
])

# Compile the model with appropriate loss and metrics
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3)

model.summary()

# Train the model
model.fit(x=raw_X_train, y=raw_y_train, validation_data=(raw_X_val, raw_y_val), epochs=20, batch_size=32,
          callbacks=[
              WandbMetricsLogger(log_freq=5),
              WandbModelCheckpoint("models"),
              callback
          ], )

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(raw_X_test, raw_y_test)
print('Test accuracy:', test_acc)

model.save("model.h5")


datagen = ImageDataGenerator(rotation_range=20,
                             featurewise_center=False,
                             featurewise_std_normalization=False,
                             width_shift_range=0.1,
                             height_shift_range=0.1,
                             horizontal_flip=True,
                             zca_whitening=False)


datagen = ImageDataGenerator(
    rotation_range=45,
    zoom_range=0.3,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.25,
    fill_mode="nearest",
)


# Perform data augmentation on X
X_augmented = []
for batch in datagen.flow(X, batch_size=X.shape[0], shuffle=False):
    X_augmented.append(batch)
    if len(X_augmented) == 1:
        break

# 2. Split the processed data into TRAIN/VAL/TEST sets

# Split data into train and test sets
X_train_val, X_test, y_train_val, y_test = train_test_split(X_augmented[0], y_encoded, test_size=0.1, random_state=42,
                                                            stratify=y_encoded)
# Split train_val data into train and validation sets
# NO LONGER NEEDED IN SPLIT 3!
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42,
                                                  stratify=y_train_val)
