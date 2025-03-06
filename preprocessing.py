import os
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Path to dataset
data_dir = "asl_alphabet_train/asl_alphabet_train"

# Define image size & categories (A-Z)
img_size = 64
labels = sorted(os.listdir(data_dir))  # ['A', 'B', 'C', ..., 'Z']

X, y = [], []

# Read images and resize them
for label in labels:
    folder_path = os.path.join(data_dir, label)
    for img in os.listdir(folder_path):
        img_array = cv2.imread(os.path.join(folder_path, img))
        img_array = cv2.resize(img_array, (img_size, img_size))
        X.append(img_array)
        y.append(labels.index(label))  # Convert label to a number

# Convert to NumPy arrays
X = np.array(X) / 255.0  # Normalize
y = to_categorical(np.array(y), num_classes=26)  # One-hot encode labels

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
