import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2, preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

# --- CONFIG ---
IMG_SIZE = 224
BATCH_SIZE = 32
CLASSES = 26
EPOCHS = 30
FINE_TUNE_EPOCHS = 15
DATA_PATH = "path_to_your_dataset"
SEED = 123
tf.random.set_seed(SEED)

# --- DATA PIPELINE ---
def create_dataset(data_path, img_size, batch_size, seed):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_path,
        validation_split=0.2,
        subset="training",
        seed=seed,
        image_size=(img_size, img_size),
        batch_size=batch_size,
        label_mode="categorical",
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_path,
        validation_split=0.2,
        subset="validation",
        seed=seed,
        image_size=(img_size, img_size),
        batch_size=batch_size,
        label_mode="categorical",
    )

    # Data augmentation (helps generalization)
    aug = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1)
    ])

    def preprocess(x, y):
        x = aug(x)
        return preprocess_input(x), y

    train_ds = train_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=tf.data.AUTOTUNE)

    return train_ds.prefetch(tf.data.AUTOTUNE), val_ds.prefetch(tf.data.AUTOTUNE)

train_ds, val_ds = create_dataset(DATA_PATH, IMG_SIZE, BATCH_SIZE, SEED)

# --- MODEL BUILDER ---
def build_model(trainable=False, fine_tune_at=None):
    base = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet"
    )

    if trainable:
        base.trainable = True
        if fine_tune_at is not None:
            for layer in base.layers[:fine_tune_at]:
                layer.trainable = False
    else:
        base.trainable = False

    model = Sequential([
        base,
        GlobalAveragePooling2D(),
        Dropout(0.4),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(CLASSES, activation='softmax')
    ])

    return model

# --- CALLBACKS ---
callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1),
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
    ModelCheckpoint("best_mobilenetv2.h5", monitor="val_loss", save_best_only=True, verbose=1)
]

# --- PHASE 1: FEATURE EXTRACTION ---
model = build_model(trainable=False)
model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)

history_1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

# --- PHASE 2: FINE-TUNING ---
model = build_model(trainable=True, fine_tune_at=100)
model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
    metrics=['accuracy']
)

history_2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=FINE_TUNE_EPOCHS,
    callbacks=callbacks
)

# --- MERGE HISTORIES ---
acc = history_1.history['accuracy'] + history_2.history['accuracy']
val_acc = history_1.history['val_accuracy'] + history_2.history['val_accuracy']

# --- PLOT RESULTS ---
plt.figure(figsize=(8,5))
plt.plot(acc, 'o-', label='Train Accuracy')
plt.plot(val_acc, 'x-', label='Validation Accuracy')
plt.title("Model Accuracy Progression")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
