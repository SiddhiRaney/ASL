import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2, preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

# --- Config ---
IMG_SIZE, BATCH, CLASSES, EPOCHS, FT_EPOCHS = 224, 32, 26, 50, 20
DATA_PATH, SEED = "path_to_your_dataset", 123
tf.random.set_seed(SEED)

# --- Dataset Loader ---
load_ds = lambda split: tf.keras.utils.image_dataset_from_directory(
    DATA_PATH, validation_split=0.2, subset=split, seed=SEED,
    image_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH, label_mode='categorical'
)
prep_ds = lambda ds: ds.map(lambda x, y: (preprocess_input(x), y)).prefetch(tf.data.AUTOTUNE)

train_ds, val_ds = prep_ds(load_ds("training").shuffle(1000)), prep_ds(load_ds("validation"))

# --- Build Model ---
def build_model(base_trainable=False, fine_tune_from=None):
    base = MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights='imagenet')
    base.trainable = base_trainable
    if base_trainable and fine_tune_from is not None:
        for l in base.layers[:fine_tune_from]: l.trainable = False
    return Sequential([
        base, GlobalAveragePooling2D(), Dropout(0.5),
        Dense(128, activation='relu'), Dropout(0.3),
        Dense(CLASSES, activation='softmax')
    ])

model = build_model()
model.compile(optimizer=Adam(1e-3), loss=CategoricalCrossentropy(label_smoothing=0.1), metrics=['accuracy'])

# --- Callbacks ---
cb = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=0),
    EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=0),
    ModelCheckpoint("best_model.h5", save_best_only=True, monitor="val_loss", verbose=0)
]

# --- Train ---
hist1 = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=cb)

# --- Fine-tune ---
model = build_model(base_trainable=True, fine_tune_from=-30)
model.compile(optimizer=Adam(1e-5), loss=CategoricalCrossentropy(label_smoothing=0.1), metrics=['accuracy'])
hist2 = model.fit(train_ds, validation_data=val_ds, epochs=FT_EPOCHS, callbacks=cb)

# --- Plot ---
acc = hist1.history['accuracy'] + hist2.history['accuracy']
val_acc = hist1.history['val_accuracy'] + hist2.history['val_accuracy']
plt.plot(acc, label='Train Acc', marker='o')
plt.plot(val_acc, label='Val Acc', marker='x')
plt.title('Accuracy Over Epochs'); plt.xlabel('Epoch'); plt.ylabel('Accuracy')
plt.grid(); plt.legend(); plt.tight_layout(); plt.show()
