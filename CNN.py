import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2, preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

# --- CONFIG ---
sz = 224
bs = 32
cls = 26
e1 = 30
e2 = 15
dpath = "path_to_your_dataset"
seed = 123
tf.random.set_seed(seed)

# --- DATA PIPELINE ---
def make_ds(path, img_sz, batch, sd):
    ds_tr = tf.keras.utils.image_dataset_from_directory(
        path, validation_split=0.2, subset="training", seed=sd,
        image_size=(img_sz, img_sz), batch_size=batch, label_mode="categorical"
    )
    ds_val = tf.keras.utils.image_dataset_from_directory(
        path, validation_split=0.2, subset="validation", seed=sd,
        image_size=(img_sz, img_sz), batch_size=batch, label_mode="categorical"
    )

    aug = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1)
    ])

    auto = tf.data.AUTOTUNE

    ds_tr = (
        ds_tr.map(lambda x, y: (preprocess_input(aug(x)), y), num_parallel_calls=auto)
             .cache().shuffle(1000).prefetch(auto)
    )
    ds_val = (
        ds_val.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=auto)
               .cache().prefetch(auto)
    )
    return ds_tr, ds_val

tr, val = make_ds(dpath, sz, bs, seed)

# --- MODEL ---
def build(trainable=False, ft_at=None):
    base = MobileNetV2(input_shape=(sz, sz, 3), include_top=False, weights="imagenet")
    base.trainable = trainable

    if trainable and ft_at:
        for l in base.layers[:ft_at]:
            l.trainable = False

    return Sequential([
        base,
        GlobalAveragePooling2D(),
        Dropout(0.4),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(cls, activation='softmax')
    ])

# --- CALLBACKS ---
cbs = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1),
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
    ModelCheckpoint("best_mn2.h5", monitor="val_loss", save_best_only=True, verbose=1)
]

# --- PHASE 1 ---
m1 = build(trainable=False)
m1.compile(optimizer=Adam(1e-3),
           loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
           metrics=['accuracy'])

h1 = m1.fit(tr, validation_data=val, epochs=e1, callbacks=cbs, verbose=1)

# --- PHASE 2 ---
m2 = build(trainable=True, fine_tune_at=100)
m2.set_weights(m1.get_weights())

m2.compile(optimizer=Adam(1e-5),
           loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
           metrics=['accuracy'])

h2 = m2.fit(tr, validation_data=val, epochs=e2, callbacks=cbs, verbose=1)

# --- MERGE METRICS ---
acc = h1.history['accuracy'] + h2.history['accuracy']
v_acc = h1.history['val_accuracy'] + h2.history['val_accuracy']

# --- PLOT ---
plt.figure(figsize=(8,5))
plt.plot(acc, 'o-', label='Train Acc')
plt.plot(v_acc, 'x-', label='Val Acc')
plt.title("MobileNetV2 Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
