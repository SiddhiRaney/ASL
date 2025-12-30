import os
from datetime import datetime
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
)

# ===================== GLOBAL OPTIMIZATIONS ===================== #

# Enable mixed precision (GPU only)
if tf.config.list_physical_devices("GPU"):
    tf.keras.mixed_precision.set_global_policy("mixed_float16")

# Enable XLA
tf.config.optimizer.set_jit(True)

# Optional reproducibility
tf.random.set_seed(42)

AUTOTUNE = tf.data.AUTOTUNE

# ===================== PATHS ===================== #

ts = datetime.now().strftime("%Y%m%d_%H%M%S")

ck_dir = "checkpoints"
lg_dir = f"logs/train_{ts}"

os.makedirs(ck_dir, exist_ok=True)
os.makedirs(lg_dir, exist_ok=True)

ck_path = os.path.join(ck_dir, f"sign_{ts}.keras")

# ===================== CALLBACKS ===================== #

callbacks = [
    ModelCheckpoint(
        filepath=ck_path,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    ),
    EarlyStopping(
        monitor="val_loss",
        patience=7,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=3,
        min_lr=1e-6,
        verbose=1
    ),
    TensorBoard(
        log_dir=lg_dir,
        histogram_freq=0,     # Much faster
        update_freq="epoch"
    )
]

# ===================== TRAIN FUNCTION ===================== #

def train(model, x_tr, y_tr, x_val, y_val, epochs=40, batch_size=64):

    tr_ds = (
        tf.data.Dataset.from_tensor_slices((x_tr, y_tr))
        .shuffle(buffer_size=min(len(x_tr), 10_000))
        .batch(batch_size)
        .cache()
        .prefetch(AUTOTUNE)
    )

    val_ds = (
        tf.data.Dataset.from_tensor_slices((x_val, y_val))
        .batch(batch_size)
        .cache()
        .prefetch(AUTOTUNE)
    )

    history = model.fit(
        tr_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=2
    )

    return history

# ===================== PLOT ===================== #

def plot(history):
    metrics = [("accuracy", "Accuracy"), ("loss", "Loss")]
    plt.figure(figsize=(12, 5))

    for i, (m, title) in enumerate(metrics):
        plt.subplot(1, 2, i + 1)
        plt.plot(history.history[m], label="Train")
        plt.plot(history.history[f"val_{m}"], label="Validation")
        plt.title(title)
        plt.xlabel("Epochs")
        plt.ylabel(title)
        plt.legend()
        plt.grid(alpha=0.4)

    plt.tight_layout()
    plt.show()

# ===================== RUN ===================== #

history = train(model, X_train, y_train, X_test, y_test)
plot(history)
