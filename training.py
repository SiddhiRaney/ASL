import os
import matplotlib.pyplot as plt
from datetime import datetime
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
)


# === Setup Directories === #
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
ckpt_dir = "checkpoints"
log_dir = f"logs/train_{timestamp}"
os.makedirs(ckpt_dir, exist_ok=True)
ckpt_path = f"{ckpt_dir}/sign_lang_{timestamp}.h5"


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
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
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


# === Model Training === #
def train_model(model, X_train, y_train, X_val, y_val, epochs=30, batch_size=64):
    """Train the model with callbacks and optimized parameters."""
    return model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        callbacks=get_callbacks(),
        verbose=2
    )


# === Plot Metrics === #
def plot_metrics(history):
    """Plot training and validation accuracy/loss."""
    plt.figure(figsize=(12, 5))
    for i, (metric, label, title) in enumerate([
        ('accuracy', 'Accuracy', 'Model Accuracy'),
        ('loss', 'Loss', 'Model Loss')
    ]):
        plt.subplot(1, 2, i + 1)
        plt.plot(history.history[metric], 'o-', label=f'Train {label}')
        plt.plot(history.history[f'val_{metric}'], 'o-', label=f'Val {label}')
        plt.title(title)
        plt.xlabel('Epoch')
        plt.ylabel(label)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()


# === Execute Training and Plot === #
history = train_model(model, X_train, y_train, X_test, y_test)
plot_metrics(history)
