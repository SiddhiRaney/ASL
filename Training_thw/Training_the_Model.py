from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

checkpoint = ModelCheckpoint(
    "sign_language_model.h5",
    monitor="val_loss",
    save_best_only=True,
    mode="auto",          # auto-detect minimize/maximize
    verbose=1,
    save_weights_only=False,  # set True if you want smaller files with weights only
    save_freq='epoch'     # explicitly save every epoch
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,           # stops training if no improvement after 3 epochs
    restore_best_weights=True,
    verbose=1
)

history = model.fit(
    X_train, y_train,
    epochs=10,
    validation_data=(X_test, y_test),
    batch_size=32,
    callbacks=[checkpoint, early_stop]
)
