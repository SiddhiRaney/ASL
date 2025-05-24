from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from pathlib import Path

# Create a directory to store model checkpoints (if it doesn't exist)
checkpoint_dir = Path("checkpoints")
checkpoint_dir.mkdir(parents=True, exist_ok=True)

# Callback: Save the best model based on validation accuracy
checkpoint_cb = ModelCheckpoint(
    filepath=checkpoint_dir / "sign_language_model.h5",  # Save path
    monitor="val_accuracy",  # Monitor validation accuracy
    save_best_only=True,     # Save only the best model
    verbose=1
)

# Callback: Stop training early if validation loss doesn't improve
early_stop_cb = EarlyStopping(
    monitor="val_loss",     # Monitor validation loss
    patience=3,             # Wait 3 epochs before stopping
    restore_best_weights=True,  # Restore weights from best epoch
    verbose=1
)

# Callback: Reduce learning rate if validation loss plateaus
reduce_lr_cb = ReduceLROnPlateau(
    monitor="val_loss",     # Monitor validation loss
    factor=0.2,             # Reduce LR by a factor of 0.2
    patience=2,             # Wait 2 epochs before reducing
    min_lr=1e-6,            # Do not reduce below this LR
    verbose=1
)

# Train the model with callbacks
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=20,
    batch_size=32,
    shuffle=True,
    callbacks=[checkpoint_cb, early_stop_cb, reduce_lr_cb],
    verbose=2  # Print one line per epoch
)

# Function to plot training and validation accuracy/loss
def plot_history(history):
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    plt.figure(figsize=(14, 5))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(acc, label="Train Accuracy")
    plt.plot(val_acc, label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Train vs Validation Accuracy")
    plt.legend()
    plt.grid(True)

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(loss, label="Train Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Train vs Validation Loss")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# Visualize the training history
plot_history(history)
