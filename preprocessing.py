import tensorflow as tf

# ——— Mixed precision (if you have a compatible GPU) ———
# from tensorflow.keras import mixed_precision
# mixed_precision.set_global_policy("mixed_float16")  # Set global policy to mixed precision for faster compute on supported hardware

# Path to the directory containing ASL alphabet images
DATA_DIR    = "asl_alphabet_train/asl_alphabet_train"
# Target image dimensions for resizing
IMG_SIZE    = (64, 64)
# Number of samples per batch
BATCH_SIZE  = 32
# Fraction of data to reserve for validation
VALID_SPLIT = 0.2
# Seed for reproducible shuffling and splitting
SEED        = 42

# ——— Build a reusable data augmentation pipeline ———
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),   # Randomly flip images horizontally
    tf.keras.layers.RandomRotation(0.1),         # Randomly rotate images by up to 10%
    tf.keras.layers.RandomZoom(0.1),             # Randomly zoom in/out by up to 10%
], name="data_augmentation")

# ——— Build a resize and rescale pipeline ———
rescale_and_resize = tf.keras.Sequential([
    tf.keras.layers.Resizing(*IMG_SIZE),      # Resize images to (64, 64)
    tf.keras.layers.Rescaling(1./255),        # Scale pixel values to [0, 1]
], name="resize_and_rescale")

# ——— Helper function to create and optimize datasets ———
def make_dataset(subset):
    # Load images from directory, automatically split into training/validation
    ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        labels="inferred",                # Infer labels from subdirectory names
        label_mode="categorical",          # One-hot encode labels
        validation_split=VALID_SPLIT,        # Use VALID_SPLIT for validation
        subset=subset,                       # 'training' or 'validation'
        seed=SEED,                           # Seed to ensure consistent split
        image_size=IMG_SIZE,                 # Resize during load
        batch_size=BATCH_SIZE,               # Batch size
    )
    return (
        ds
        .shuffle(1000, seed=SEED)           # Shuffle dataset before caching
        .cache()                             # Cache data to improve performance
        .map(lambda x, y: (x / 255.0, y), num_parallel_calls=tf.data.AUTOTUNE)
                                            # Additional normalization and parallel mapping
        .prefetch(tf.data.AUTOTUNE)         # Prefetch batches for smoother training
    )

# Create training and validation datasets
train_ds = make_dataset("training")
val_ds   = make_dataset("validation")

# ——— Define the CNN model architecture ———
model = tf.keras.Sequential([
    data_augmentation,                    # Apply data augmentation during training
    rescale_and_resize,                   # Ensure consistent image size and scaling

    # Convolutional block 1
    tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(),       # Downsample feature maps

    # Convolutional block 2
    tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(),

    # Convolutional block 3
    tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(),

    # Flatten feature maps to feed into dense layers
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),  # Fully connected layer
    tf.keras.layers.Dropout(0.5),                   # Dropout for regularization
    tf.keras.layers.Dense(26, activation='softmax'),# Output layer for 26 classes
], name="asl_cnn")

# Compile the model with optimizer, loss, and metrics
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),  # Low learning rate for stable training
    loss="categorical_crossentropy",          # Suitable for multi-class classification
    metrics=["accuracy"],                     # Track accuracy during training
)

# ——— Callbacks to improve training process ———
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True
    ),  # Stop training if validation loss doesn't improve for 3 epochs
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=2
    ),  # Reduce learning rate if validation loss plateaus
]

# ——— Train the model ———
history = model.fit(
    train_ds,                  # Training dataset
    validation_data=val_ds,    # Validation dataset
    epochs=20,                 # Maximum number of epochs
    callbacks=callbacks,       # Callbacks for early stopping and LR reduction
)  # Returns training history object
