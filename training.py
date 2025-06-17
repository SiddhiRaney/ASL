import os
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from datetime import datetime

# Create checkpoints directory if not exists
os.makedirs("checkpoints", exist_ok=True)

# Timestamped filename for better versioning
model_checkpoint_filename = f"checkpoints/sign_language_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5"

# Callbacks configuration
training_callbacks = [
    ModelCheckpoint(
        filepath=model_checkpoint_filename,
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
training_history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=20,
    batch_size=32,
    shuffle=True,
    callbacks=training_callbacks,
    verbose=2
)

# Plot training history
def plot_training_metrics(history_data):
    plt.figure(figsize=(12, 5))

    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(history_data.history['accuracy'], label='Train Accuracy')
    plt.plot(history_data.history['val_accuracy'], label='Validation Accuracy')
    plt.title("Model Accuracy Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(history_data.history['loss'], label='Train Loss')
    plt.plot(history_data.history['val_loss'], label='Validation Loss')
    plt.title("Model Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

plot_training_metrics(training_history)
