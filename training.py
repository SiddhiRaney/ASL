import os
from datetime import datetime
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
)


# === Paths === #
ts = datetime.now().strftime('%Y%m%d_%H%M%S')
ck_dir = "checkpoints"
lg_dir = f"logs/train_{ts}"

os.makedirs(ck_dir, exist_ok=True)
os.makedirs(lg_dir, exist_ok=True)

ck_path = os.path.join(ck_dir, f"sign_{ts}.h5")


# === Callbacks === #
def cb():
    return [
        ModelCheckpoint(
            filepath=ck_path,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
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
            histogram_freq=1,
            profile_batch=0
        )
    ]


# === Training Function === #
def train(model, x_tr, y_tr, x_val, y_val, ep=40, bs=64):

    # Mixed precision (GPU only)
    if tf.config.list_physical_devices('GPU'):
        tf.keras.mixed_precision.set_global_policy('mixed_float16')

    # TF Dataset for speed
    tr_ds = (
        tf.data.Dataset.from_tensor_slices((x_tr, y_tr))
        .shuffle(len(x_tr))
        .batch(bs)
        .prefetch(tf.data.AUTOTUNE)
    )

    val_ds = (
        tf.data.Dataset.from_tensor_slices((x_val, y_val))
        .batch(bs)
        .prefetch(tf.data.AUTOTUNE)
    )

    hist = model.fit(
        tr_ds,
        validation_data=val_ds,
        epochs=ep,
        callbacks=cb(),
        verbose=2
    )

    return hist


# === Plot === #
def plot(hist):
    mets = [('accuracy', 'Accuracy'), ('loss', 'Loss')]
    plt.figure(figsize=(12, 5))

    for i, (m, lbl) in enumerate(mets):
        plt.subplot(1, 2, i + 1)
        plt.plot(hist.history[m], 'o-', label=f'Train {lbl}')
        plt.plot(hist.history[f'val_{m}'], 'o-', label=f'Val {lbl}')
        plt.title(lbl)
        plt.xlabel('Epochs')
        plt.ylabel(lbl)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()


# === Run === #
history = train(model, X_train, y_train, X_test, y_test)
plot(history)
