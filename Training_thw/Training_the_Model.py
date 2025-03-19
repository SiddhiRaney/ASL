from tensorflow.keras.callbacks import ModelCheckpoint

# Define ModelCheckpoint to save the best model during training
checkpoint = ModelCheckpoint("sign_language_model.h5", 
                              monitor="val_loss", 
                              save_best_only=True, 
                              mode="min", 
                              verbose=1)

# Train the model and save the best version based on validation loss
history = model.fit(X_train, y_train, 
                    epochs=10, 
                    validation_data=(X_test, y_test), 
                    batch_size=32, 
                    callbacks=[checkpoint])
