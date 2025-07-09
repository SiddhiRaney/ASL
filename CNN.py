import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.applications import MobileNetV2, preprocess_input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import os

# --- Config ---
IMG_SIZE, BATCH_SIZE, NUM_CLASSES = 224, 32, 26
EPOCHS = 50
DATA_PATH = "path_to_your_dataset"
SEED = 123

# Set seed for reproducibility
tf.random.set_seed(SEED)

# --- Load Data ---
def load_dataset(split):
    return tf.keras.utils.image_dataset_from_directory(
        DATA_PATH,
        validation_split=0.2,
        subset=split,
        seed=SEED,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode='categorical'
    )

train_ds = load_dataset("training")
val_ds = load_dataset("validation")

AUTOTUNE = tf.data.AUTOTUNE
def preprocess(ds):
    return ds.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

train_ds = preprocess(train_ds.shuffle(1000))
val_ds = preprocess(val_ds)

# --- Model ---
base_model = MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights='imagenet')
base_model.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(NUM_CLASSES, activation='softmax')
])

model.compile(
    optimizer=Adam(1e-3),
    loss=CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)

# --- Callbacks ---
callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
    EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=1),
    ModelCheckpoint("best_model.h5", monitor="val_loss", save_best_only=True, verbose=1)
]

# --- Train Base Model ---
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

# --- Fine-tune ---
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=Adam(1e-5),
    loss=CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)

ft_history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,
    callbacks=callbacks
)

# --- Plot ---
acc = history.history['accuracy'] + ft_history.history['accuracy']
val_acc = history.history['val_accuracy'] + ft_history.history['val_accuracy']

plt.figure(figsize=(10, 5))
plt.plot(acc, label='Train Accuracy', marker='o')
plt.plot(val_acc, label='Validation Accuracy', marker='x')
plt.title('Training & Validation Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
