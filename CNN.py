import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, GlobalAveragePooling2D,
    Dense, Dropout, BatchNormalization
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

# 1. Data augmentation pipeline
from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1,
    validation_split=0.2
)

# 2. Build deeper, normalized model
model = Sequential([
    # Block 1
    Conv2D(32, 3, activation='relu', kernel_regularizer=l2(1e-4),
           input_shape=(img_size, img_size, 3)),  # Input layer
    BatchNormalization(),                      # Normalize activations
    MaxPooling2D(2),
    
    # Block 2
    Conv2D(64, 3, activation='relu', kernel_regularizer=l2(1e-4)),
    BatchNormalization(),
    MaxPooling2D(2),
    
    # Block 3
    Conv2D(128, 3, activation='relu', kernel_regularizer=l2(1e-4)),
    BatchNormalization(),
    MaxPooling2D(2),
    
    # Global pooling instead of flatten
    GlobalAveragePooling2D(),
    
    Dense(256, activation='relu'),             # Dense head
    Dropout(0.5),                              # Dropout for regularization
    Dense(26, activation='softmax')            # 26-way output
])

# 3. Compile with a somewhat lower initial LR
optimizer = Adam(learning_rate=1e-3)
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 4. Callbacks for LR scheduling & early stop
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=3, verbose=1
)
early_stop = EarlyStopping(
    monitor='val_loss', patience=7, restore_best_weights=True, verbose=1
)

model.summary()

# 5. Fit using generators
batch_size = 32
train_flow = train_gen.flow_from_directory(
    train_dir, target_size=(img_size, img_size),
    batch_size=batch_size, class_mode='categorical', subset='training'
)
val_flow = train_gen.flow_from_directory(
    train_dir, target_size=(img_size, img_size),
    batch_size=batch_size, class_mode='categorical', subset='validation'
)

history = model.fit(
    train_flow,
    validation_data=val_flow,
    epochs=50,
    callbacks=[reduce_lr, early_stop]
)
