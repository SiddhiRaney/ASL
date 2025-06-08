from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from pathlib import Path

# Create checkpoints directory
Path("checkpoints").mkdir(exist_ok=True)

# Define training callbacks
callbacks = [
    ModelCheckpoint(
        "checkpoints/sign_language_model.h5",
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    ),
    EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )
]

# Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=20,
    batch_size=32,
    shuffle=True,
    callbacks=callbacks,
    verbose=2
)

# Plot training history
def plot_history(history):
    metrics = history.history
    epochs = range(1, len(metrics["accuracy"]) + 1)

    plt.figure(figsize=(14, 5))

    for i, metric in enumerate(["accuracy", "loss"], 1):
        plt.subplot(1, 2, i)
        plt.plot(epochs, metrics[metric], label="Train")
        plt.plot(epochs, metrics[f"val_{metric}"], label="Val")
        plt.title(metric.capitalize())
        plt.xlabel("Epochs")
        plt.ylabel(metric.capitalize())
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.show()

plot_history(history)
