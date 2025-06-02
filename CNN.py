import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# --- Configuration ---
data_dir   = train_dir   # Path to your training images
img_size   = 224         # Height and width for each input image
bs         = 32          # Batch size
n_classes  = 26          # Number of output classes (e.g., A–Z)
weight_decay = 1e-4      # L2 regularization factor
initial_lr   = 1e-3      # Initial learning rate

# 1. Data augmentation + normalization
datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,   # Normalize pixel values
    rotation_range=15,     # Rotate ±15°
    width_shift_range=0.1, # Shift horizontally ±10%
    height_shift_range=0.1,# Shift vertically ±10%
    horizontal_flip=True,  # Random horizontal flip
    zoom_range=0.1,        # Zoom in/out ±10%
    validation_split=0.2   # Split 20% of data for validation
)

# 2. Build CNN model
model = Sequential([
    # Block 1
    Conv2D(
        filters=32,
        kernel_size=3,
        activation='relu',
        kernel_regularizer=l2(weight_decay),
        kernel_initializer='he_normal',
        input_shape=(img_size, img_size, 3)
    ),
    BatchNormalization(),
    MaxPooling2D(pool_size=2),

    # Block 2
    Conv2D(
        filters=64,
        kernel_size=3,
        activation='relu',
        kernel_regularizer=l2(weight_decay),
        kernel_initializer='he_normal'
    ),
    BatchNormalization(),
    MaxPooling2D(pool_size=2),

    # Block 3
    Conv2D(
        filters=128,
        kernel_size=3,
        activation='relu',
        kernel_regularizer=l2(weight_decay),
        kernel_initializer='he_normal'
    ),
    BatchNormalization(),
    MaxPooling2D(pool_size=2),

    # Global pooling + classifier head
    GlobalAveragePooling2D(),
    Dense(
        units=256,
        activation='relu',
        kernel_regularizer=l2(weight_decay),
        kernel_initializer='he_normal'
    ),
    Dropout(0.5),
    Dense(units=n_classes, activation='softmax')
])

# 3. Compile model
opt = Adam(learning_rate=initial_lr)
model.compile(
    optimizer=opt,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# 4. Callbacks
lr_reduce = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    verbose=1
)
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=7,
    restore_best_weights=True,
    verbose=1
)

# 5. Data generators for training/validation
train_flow = datagen.flow_from_directory(
    data_dir,
    target_size=(img_size, img_size),
    batch_size=bs,
    class_mode='categorical',
    subset='training'
)
val_flow = datagen.flow_from_directory(
    data_dir,
    target_size=(img_size, img_size),
    batch_size=bs,
    class_mode='categorical',
    subset='validation'
)

# 6. Train
history = model.fit(
    train_flow,
    validation_data=val_flow,
    epochs=50,
    callbacks=[lr_reduce, early_stop]
)
