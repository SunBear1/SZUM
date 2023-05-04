import io
import json
import threading

import numpy as np
import tensorflow as tf
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint

import wandb

NUMBER_OF_CLASSES = 5
dtype = 'float32'

data_sets_split_1 = {}
labels_split_1 = ["raw_x_train", "raw_y_train", "raw_x_val", "raw_y_val", "raw_x_test", "raw_y_test"]
data_sets_split_2 = {}
labels_split_2 = ["x_train", "y_train", "x_val", "y_val", "x_test", "y_test"]
data_sets_split_3 = {}
labels_split_3 = ["x_train_val", "y_train_val", "x_test", "y_test"]


def train_with_effnet_on_raw_data(x_train_set, y_train_set, x_val_set, y_val_set, x_test_set, y_test_set):
    wandb.init(project="SZUM")

    print("aligning data.. ")

    print("started_training..")

    # Define the model architecture
    model = tf.keras.Sequential([
        tf.keras.applications.MobileNetV3Large(input_shape=(224, 224, 3), include_top=False,
                                                              weights='imagenet'),
        # tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(NUMBER_OF_CLASSES, activation='softmax')
    ])

    # Compile the model with appropriate loss and metrics
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8)

    model.summary()

    # Train the model
    model.fit(x=x_train_set, y=y_train_set, validation_data=(x_val_set, y_val_set), epochs=5, batch_size=16,
              callbacks=[
                  WandbMetricsLogger(log_freq=5),
                  WandbModelCheckpoint("models"),
                  callback
              ], )

    # Evaluate the model on the test set
    test_loss, test_acc = model.evaluate(x_test_set, y_test_set)
    print('Test accuracy:', test_acc)

    model.save("model.h5")


def load_data(label: str, split: str, data_set: dict):
    with open(f"../data/{split}/{label}.json", "r") as infile:
        serialized_data = infile.read()
        serialized = json.loads(serialized_data)
        memfile = io.BytesIO(serialized.encode('latin-1'))
        data = np.load(memfile, allow_pickle=True)
        data_set[label] = data.astype(dtype)
        print(f"{label} loaded.")


threads = []
for set_label in labels_split_1:
    t = threading.Thread(target=load_data, args=(set_label, "split_1", data_sets_split_1))
    threads.append(t)
# for set_label in labels_split_2:
#     t = threading.Thread(target=load_data, args=(set_label, "split_2", data_sets_split_2))
#     threads.append(t)
# for set_label in labels_split_3:
#     t = threading.Thread(target=load_data, args=(set_label, "split_3"))
#     threads.append(t)
for t in threads:
    t.start()

for t in threads:
    t.join()

data_sets_split_1["raw_y_train"] = tf.keras.utils.to_categorical(data_sets_split_1["raw_y_train"],
                                                                 num_classes=NUMBER_OF_CLASSES)
data_sets_split_1["raw_y_val"] = tf.keras.utils.to_categorical(data_sets_split_1["raw_y_val"],
                                                               num_classes=NUMBER_OF_CLASSES)
data_sets_split_1["raw_y_test"] = tf.keras.utils.to_categorical(data_sets_split_1["raw_y_test"],
                                                                num_classes=NUMBER_OF_CLASSES)

train_with_effnet_on_raw_data(x_train_set=data_sets_split_1["raw_x_train"],
                              y_train_set=data_sets_split_1["raw_y_train"],
                              x_val_set=data_sets_split_1["raw_x_val"],
                              y_val_set=data_sets_split_1["raw_y_val"],
                              x_test_set=data_sets_split_1["raw_x_test"],
                              y_test_set=data_sets_split_1["raw_y_test"]
                              )
