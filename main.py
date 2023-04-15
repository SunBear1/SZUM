import os
import random

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from functions import present_sample_images, present_augmented_data, present_first_dataset_split

data_dir = "dataset"
labels = os.listdir(data_dir)

images = []
for label in labels:
    label_dir = os.path.join(data_dir, label)
    for img_file in os.listdir(label_dir):
        img_path = os.path.join(label_dir, img_file)
        img = Image.open(img_path)
        img_copy = img.copy()
        img.close()
        img_copy = img_copy.resize(size=(224, 224))
        images.append((img_copy, label))

present_sample_images(images=images)

X = []
y = []

for image, label in images:
    if np.array(image).shape == (224, 224, 3):
        X.append(np.array(image))
        y.append(label)

X = np.array(X)
y = np.array(y)

# Normalize the data
X = X.astype('float32')
X /= 255.0

le = LabelEncoder()
y_encoded = le.fit_transform(y)

label_dict = dict(zip(le.classes_, le.transform(le.classes_)))
print(label_dict)

datagen = ImageDataGenerator(rotation_range=20,
                             featurewise_center=False,
                             featurewise_std_normalization=False,
                             width_shift_range=0.1,
                             height_shift_range=0.1,
                             horizontal_flip=True,
                             zca_whitening=False)

# Perform data augmentation on X
X_augmented = []
for batch in datagen.flow(X, batch_size=X.shape[0], shuffle=False):
    X_augmented.append(batch)
    if len(X_augmented) == 1:
        break

present_augmented_data(X=X, X_augmented=X_augmented)

random.seed(42)

# 1. Split the data into TRAIN/VAL/TEST sets
random.shuffle(images)
train_split = 0.7  # 70% for training
val_split = 0.2  # 20% for validation
test_split = 0.1  # 10% for testing

num_train = int(train_split * len(images))
num_val = int(val_split * len(images))

raw_X_train = np.array([img[0] for img in images[:num_train]])
raw_y_train = np.array([img[1] for img in images[:num_train]])

raw_X_val = np.array([img[0] for img in images[num_train:num_train + num_val]])
raw_y_val = np.array([img[1] for img in images[num_train:num_train + num_val]])

raw_X_test = np.array([img[0] for img in images[num_train + num_val:]])
raw_y_test = np.array([img[1] for img in images[num_train + num_val:]])

present_first_dataset_split(label_dict=label_dict, y_train=raw_y_train, y_val=raw_y_val, y_test=raw_y_test)

# 2. Split the processed data into TRAIN/VAL/TEST sets

# Split data into train and test sets
X_train_val, X_test, y_train_val, y_test = train_test_split(X_augmented[0], y_encoded, test_size=0.1, random_state=42,
                                                            stratify=y_encoded)
# Split train_val data into train and validation sets
# NO LONGER NEEDED IN SPLIT 3!
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42,
                                                  stratify=y_train_val)

# Count number of samples in each class for each set
num_classes = len(label_dict)
train_counts = [len(y_train[y_train == i]) for i in range(num_classes)]
val_counts = [len(y_val[y_val == i]) for i in range(num_classes)]
test_counts = [len(y_test[y_test == i]) for i in range(num_classes)]

# Plot bar chart of class distribution for each set
plt.figure(figsize=(10, 5))
plt.bar(range(num_classes), train_counts, label='Train')
plt.bar(range(num_classes), val_counts, bottom=train_counts, label='Validation')
plt.bar(range(num_classes), test_counts, bottom=[train_counts[i] + val_counts[i] for i in range(num_classes)],
        label='Test')
plt.xticks(range(num_classes), label_dict.keys())
plt.xlabel('Class')
plt.ylabel('Count')
plt.title('Class distribution in TRAIN/VAL/TEST sets')
plt.legend()
plt.show()
