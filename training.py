import os
from datetime import datetime
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
)


# === Setup Directories === #
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
ckpt_dir = "checkpoints"
log_dir = f"logs/train_{timestamp}"

os.makedirs(ckpt_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

ckpt_path = os.path.join(ckpt_dir, f"sign_lang_{timestamp}.h5")


# === Define Callbacks === #
def get_callbacks():
    return [
        ModelCheckpoint(
            filepath=ckpt_path,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=7,                 # slightly increased for more stability
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,                 # smaller decay for finer tuning
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
        TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            profile_batch=0
        )
    ]


# === Optimized Model Training === #
def train_model(model, X_train, y_train, X_val, y_val, epochs=40, batch_size=64):
    """
    Train the model with callbacks and optimized parameters.
    """
    # Enable TensorFlow mixed precision if GPU is available
    if tf.config.list_physical_devices('GPU'):
        tf.keras.mixed_precision.set_global_policy('mixed_float16')

    # Prefetch and cache data for faster I/O
    train_data = tf.data.Dataset.from_tensor_slices((X_train, y_train)) \
        .shuffle(buffer_size=len(X_train)) \
        .batch(batch_size) \
        .prefetch(tf.data.AUTOTUNE)

    val_data = tf.data.Dataset.from_tensor_slices((X_val, y_val)) \
        .batch(batch_size) \
        .prefetch(tf.data.AUTOTUNE)

    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs,
        callbacks=get_callbacks(),
        verbose=2
    )

    return history


# === Plot Training Metrics === #
def plot_metrics(history):
    """
    Plot training and validation accuracy/loss.
    """
    metrics = [('accuracy', 'Accuracy'), ('loss', 'Loss')]
    plt.figure(figsize=(12, 5))

    for i, (metric, label) in enumerate(metrics):
        plt.subplot(1, 2, i + 1)
        plt.plot(history.history[metric], 'o-', label=f'Train {label}')
        plt.plot(history.history[f'val_{metric}'], 'o-', label=f'Validation {label}')
        plt.title(f'Model {label}')
        plt.xlabel('Epochs')
        plt.ylabel(label)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()


# === Execute Training and Plot === #
history = train_model(model, X_train, y_train, X_test, y_test)
plot_metrics(history)
