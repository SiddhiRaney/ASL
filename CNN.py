import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SeparableConv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# --- Configuration ---
data_dir     = train_dir
img_size     = 224
bs           = 32
n_classes    = 26
weight_decay = 1e-4
initial_lr   = 1e-3

# Data augmentation
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1,
    validation_split=0.2
)

# Model architecture
model = Sequential([
    # Block 1
    SeparableConv2D(32, 3, activation='relu', kernel_regularizer=l2(weight_decay),
                    kernel_initializer='he_normal', input_shape=(img_size, img_size, 3)),
    BatchNormalization(),
    MaxPooling2D(2),
    Dropout(0.2),

    # Block 2
    SeparableConv2D(64, 3, activation='relu', kernel_regularizer=l2(weight_decay),
                    kernel_initializer='he_normal'),
    BatchNormalization(),
    MaxPooling2D(2),
    Dropout(0.3),

    # Block 3
    SeparableConv2D(128, 3, activation='relu', kernel_regularizer=l2(weight_decay),
                    kernel_initializer='he_normal'),
    BatchNormalization(),
    MaxPooling2D(2),
    Dropout(0.4),

    GlobalAveragePooling2D(),
    Dense(128, activation='relu', kernel_regularizer=l2(weight_decay)),
    Dropout(0.5),
    Dense(n_classes, activation='softmax')
])

# Compile
loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
model.compile(optimizer=Adam(learning_rate=initial_lr), loss=loss_fn, metrics=['accuracy'])
model.summary()

# Callbacks
lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
early_stop = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=1)
checkpoint = ModelCheckpoint("best_model.h5", monitor="val_loss", save_best_only=True, verbose=1)

# Data generators
train_flow = datagen.flow_from_directory(
    data_dir, target_size=(img_size, img_size), batch_size=bs,
    class_mode='categorical', subset='training'
)
val_flow = datagen.flow_from_directory(
    data_dir, target_size=(img_size, img_size), batch_size=bs,
    class_mode='categorical', subset='validation'
)

# Train
history = model.fit(
    train_flow,
    validation_data=val_flow,
    epochs=50,
    callbacks=[lr_reduce, early_stop, checkpoint]
)
