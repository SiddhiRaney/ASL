import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, GlobalAveragePooling2D,
    Dense, Dropout, BatchNormalization
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set parameters
dataset_dir = train_dir  # Directory with training images
img_size = 224          # Target image size (height and width)
batch_size = 32         # Batch size for training
num_classes = 26         # Number of output classes

# 1. Data augmentation pipeline
train_gen = ImageDataGenerator(
    rescale=1./255,           # Normalize pixel values to [0, 1]
    rotation_range=15,        # Randomly rotate images by up to 15 degrees
    width_shift_range=0.1,    # Random horizontal shifts
    height_shift_range=0.1,   # Random vertical shifts
    horizontal_flip=True,     # Random horizontal flipping
    zoom_range=0.1,           # Random zoom in/out by up to 10%
    validation_split=0.2      # Reserve 20% of data for validation
)

# 2. Build a deeper, regularized model
model = Sequential([
    # Block 1: Feature extraction
    Conv2D(
        filters=32,
        kernel_size=3,
        activation='relu',
        kernel_regularizer=l2(1e-4),
        input_shape=(img_size, img_size, 3),
        kernel_initializer='he_normal'
    ),  
    BatchNormalization(),       # Normalize outputs of the conv layer
    MaxPooling2D(pool_size=2), # Reduce spatial dimensions by half

    # Block 2: Increased filters for deeper features
    Conv2D(
        filters=64,
        kernel_size=3,
        activation='relu',
        kernel_regularizer=l2(1e-4),
        kernel_initializer='he_normal'
    ),
    BatchNormalization(),       # Stabilize and accelerate training
    MaxPooling2D(pool_size=2),

    # Block 3: Further increase in filter depth
    Conv2D(
        filters=128,
        kernel_size=3,
        activation='relu',
        kernel_regularizer=l2(1e-4),
        kernel_initializer='he_normal'
    ),
    BatchNormalization(),
    MaxPooling2D(pool_size=2),

    # Global average pooling to reduce parameters
    GlobalAveragePooling2D(),

    # Dense head for classification
    Dense(
        units=256,
        activation='relu',
        kernel_regularizer=l2(1e-4),
        kernel_initializer='he_normal'
    ),
    Dropout(0.5),             # Dropout for additional regularization
    Dense(units=num_classes, activation='softmax')  # Softmax for multi-class
])

# 3. Compile the model with the Adam optimizer and weight decay
optimizer = Adam(
    learning_rate=1e-3        # Initial learning rate
)
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Display model architecture
model.summary()

# 4. Configure callbacks for dynamic LR reduction and early stopping
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',      # Monitor validation loss
    factor=0.5,              # Halve the learning rate on plateau
    patience=3,              # Wait 3 epochs before reducing
    verbose=1
)
early_stop = EarlyStopping(
    monitor='val_loss',      # Stop if validation loss doesn't improve
    patience=7,              # Wait 7 epochs before stopping
    restore_best_weights=True,
    verbose=1
)

# 5. Prepare data generators for training and validation
train_flow = train_gen.flow_from_directory(
    dataset_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'         # Use training subset of data
)
val_flow = train_gen.flow_from_directory(
    dataset_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'       # Use validation subset of data
)

# 6. Train the model
history = model.fit(
    train_flow,
    validation_data=val_flow,
    epochs=50,
    callbacks=[reduce_lr, early_stop]
)
