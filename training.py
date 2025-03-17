from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt

# Define callbacks
checkpoint = ModelCheckpoint("sign_language_model.h5", monitor="val_accuracy", save_best_only=True, verbose=1)
early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True, verbose=1)

# Train the model with callbacks
history = model.fit(
    X_train, y_train, 
    epochs=20, 
    validation_data=(X_test, y_test), 
    batch_size=32, 
    callbacks=[checkpoint, early_stop]
)

# Plot training history
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Training vs Validation Accuracy")
plt.show()
