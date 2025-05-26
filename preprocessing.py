import tensorflow as tf

# ——— Mixed precision (if you have a compatible GPU) ———
# from tensorflow.keras import mixed_precision
# mixed_precision.set_global_policy("mixed_float16")

# Path to dataset
DATA_DIR    = "asl_alphabet_train/asl_alphabet_train"
IMG_SIZE    = (64, 64)
BATCH_SIZE  = 32
VALID_SPLIT = 0.2
SEED        = 42

# ——— Build a reusable preprocessing pipeline ———
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
], name="data_augmentation")

rescale_and_resize = tf.keras.Sequential([
    tf.keras.layers.Resizing(*IMG_SIZE),
    tf.keras.layers.Rescaling(1./255),
], name="resize_and_rescale")

# ——— Load datasets with caching & prefetching ———
def make_dataset(subset):
    ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        labels="inferred",
        label_mode="categorical",
        validation_split=VALID_SPLIT,
        subset=subset,
        seed=SEED,
        image_size=IMG_SIZE,     # directory loader will resize
        batch_size=BATCH_SIZE,
    )
    return (
        ds
        .shuffle(1000, seed=SEED)               # shuffle before caching so cache contains a random mix
        .cache()                                 # cache full dataset in memory/disk
        .map(lambda x, y: (x / 255.0, y), num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)             # overlap preprocess & model exec
    )

train_ds = make_dataset("training")
val_ds   = make_dataset("validation")

# ——— Model Definition ———
model = tf.keras.Sequential([
    data_augmentation,                        # apply only during training
    rescale_and_resize,                       # ensure consistent sizing & scaling
    tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),             # help regularization
    tf.keras.layers.Dense(26, activation='softmax'),
], name="asl_cnn")

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),   # lower LR often trains more stably
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

# ——— Callbacks for better training control ———
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=2
    ),
]

# ——— Train ———
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,
    callbacks=callbacks,
)
