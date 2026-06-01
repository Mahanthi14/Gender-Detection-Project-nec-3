import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Dataset path
DATASET_PATH = "dataset"

img_size = 128

X = []
y = []
labels = {}
label_count = 0

# Load dataset
for person_name in os.listdir(DATASET_PATH):
    person_path = os.path.join(DATASET_PATH, person_name)

    if not os.path.isdir(person_path):
        continue

    # Assign label
    if person_name not in labels:
        labels[person_name] = label_count
        label_count += 1

    label = labels[person_name]

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)

        img = cv2.imread(img_path)

        if img is None:
            continue

        img = cv2.resize(img, (img_size, img_size))
        img = img / 255.0  # normalization

        X.append(img)
        y.append(label)

X = np.array(X)
y = np.array(y)

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# CNN Model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(labels), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train model
model.fit(
    X_train, y_train,
    epochs=10,
    validation_data=(X_val, y_val)
)

# Save model
model.save("model/face_model.h5")

# Save labels
import json
with open("model/labels.json", "w") as f:
    json.dump(labels, f)

print("Model trained and saved successfully!")