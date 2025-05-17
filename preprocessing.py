import tensorflow as tf

# Path to dataset
data_dir = "asl_alphabet_train/asl_alphabet_train"

# Parameters
img_size   = (64, 64)
batch_size = 32
seed       = 42

# ——— Create Training & Validation Datasets ———
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    labels="inferred",
    label_mode="categorical",      # One-hot
    validation_split=0.2,
    subset="training",
    seed=seed,
    image_size=img_size,
    batch_size=batch_size,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    labels="inferred",
    label_mode="categorical",
    validation_split=0.2,
    subset="validation",
    seed=seed,
    image_size=img_size,
    batch_size=batch_size,
)

# ——— Optimize Pipeline ———
AUTOTUNE = tf.data.AUTOTUNE

def normalize(img, label):
    img = tf.cast(img, tf.float32) / 255.0
    return img, label

train_ds = (
    train_ds
    .map(normalize, num_parallel_calls=AUTOTUNE)
    .cache()                 # keep processed images in memory 
    .shuffle(1000)           # better randomness
    .prefetch(buffer_size=AUTOTUNE)
)

val_ds = (
    val_ds
    .map(normalize, num_parallel_calls=AUTOTUNE)
    .cache()
    .prefetch(buffer_size=AUTOTUNE)
)

# ——— (Optional) Data Augmentation ———
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
])

# Example of building a simple model
model = tf.keras.Sequential([
    data_augmentation,
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(26, activation='softmax'),
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Train
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
)
