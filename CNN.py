import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, GlobalAveragePooling2D,
    Dense, Dropout, BatchNormalizationimport tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, GlobalAveragePooling2D,
    Dense, Dropout, BatchNormalization
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set dataset directory and model parameters
dataset_dir = train_dir      # Path to training image directory
img_size = 224               # Target height and width for input images
batch_size = 32              # Number of samples per gradient update
num_classes = 26             # Total number of output classes (e.g., letters A-Z)

# 1. Data augmentation and normalization pipeline
train_gen = ImageDataGenerator(
    rescale=1./255,           # Scale pixel values to the [0,1] range
    rotation_range=15,        # Random rotation up to 15 degrees
    width_shift_range=0.1,    # Horizontal shift up to 10% of width
    height_shift_range=0.1,   # Vertical shift up to 10% of height
    horizontal_flip=True,     # Randomly flip inputs horizontally
    zoom_range=0.1,           # Zoom in/out by up to 10%
    validation_split=0.2      # Reserve 20% of images for validation
)

# 2. Define the CNN architecture with regularization
model = Sequential([
    # Block 1: low-level feature extraction
    Conv2D(32, 3, padding='same', activation='relu',
           kernel_regularizer=l2(1e-4), kernel_initializer='he_normal',
           input_shape=(img_size, img_size, 3)),
    BatchNormalization(),    # Normalize activations to stabilize training
    MaxPooling2D(2),         # Downsample spatial dimensions by 2

    # Block 2: deeper convolution for more complex patterns
    Conv2D(64, 3, padding='same', activation='relu',
           kernel_regularizer=l2(1e-4), kernel_initializer='he_normal'),
    BatchNormalization(),
    MaxPooling2D(2),

    # Block 3: further increase filter depth
    Conv2D(128, 3, padding='same', activation='relu',
           kernel_regularizer=l2(1e-4), kernel_initializer='he_normal'),
    BatchNormalization(),
    MaxPooling2D(2),

    # Global pooling reduces each feature map to a single value
    GlobalAveragePooling2D(),

    # Fully connected layers for classification
    Dense(256, activation='relu',
          kernel_regularizer=l2(1e-4), kernel_initializer='he_normal'),
    Dropout(0.5),            # Drop half of the units to reduce overfitting
    Dense(num_classes, activation='softmax')  # Softmax outputs class probabilities
])

# 3. Compile the model with Adam optimizer and categorical crossentropy loss
optimizer = Adam(learning_rate=1e-3)  # Base learning rate
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']               # Monitor accuracy during training
)

# Show model summary: layer shapes and parameter counts
model.summary()

# 4. Configure callbacks for dynamic learning rate and early stopping
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',      # Track validation loss to reduce LR on plateau
    factor=0.5,              # Multiply LR by 0.5 when triggered
    patience=3,              # Wait 3 epochs without improvement
    verbose=1
)
early_stop = EarlyStopping(
    monitor='val_loss',      # Track validation loss for early stop
    patience=7,              # Stop if no improvement for 7 epochs
    restore_best_weights=True,
    verbose=1
)

# 5. Prepare data generators for training and validation splits
train_flow = train_gen.flow_from_directory(
    dataset_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'        # Use the 80% training split
)
val_flow = train_gen.flow_from_directory(
    dataset_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'      # Use the 20% validation split
)

# 6. Train the model with callbacks
history = model.fit(
    train_flow,
    validation_data=val_flow,
    epochs=50,
    callbacks=[reduce_lr, early_stop]
)

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
