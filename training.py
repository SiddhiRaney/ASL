import os
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from datetime import datetime

# Ensure checkpoints directory exists
os.makedirs("checkpoints", exist_ok=True)

# Generate a timestamped filename for the best model
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
checkpoint_path = f"checkpoints/sign_language_model_{timestamp}.h5"
log_dir = f"logs/train_{timestamp}"

# Define Callbacks
callbacks = [
    ModelCheckpoint(
        filepath=checkpoint_path,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=False,
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
    ),
    TensorBoard(log_dir=log_dir, histogram_freq=1)
]

# Train the model
def train_model(model, X_train, y_train, X_test, y_test, callbacks):
    return model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=20,
        batch_size=32,
        shuffle=True,
        callbacks=callbacks,
        verbose=2
    )

# Plot accuracy and loss
def plot_training_metrics(history):
    plt.figure(figsize=(12, 5))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title("Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# Start training
training_history = train_model(model, X_train, y_train, X_test, y_test, callbacks)

# Plot metrics
plot_training_metrics(training_history)
