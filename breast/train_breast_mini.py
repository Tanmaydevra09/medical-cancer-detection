import tensorflow as tf
from tensorflow.keras import layers, models, callbacks # type: ignore
import matplotlib
matplotlib.use('Agg')   # save to file instead of popup (background process)
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# ---------------- CONFIG ----------------
IMG_SIZE    = (224, 224)
BATCH_SIZE  = 32
EPOCHS      = 5          # fewer epochs for mini dataset
FINE_TUNE_EPOCHS = 3
SEED        = 42

# ---- PATHS (mini dataset) ----
TRAIN_DIR = "mini_dataset/train"
TEST_DIR  = "mini_dataset/test"

print(f"TensorFlow Version: {tf.__version__}")

# ======================================================
# ================= CLAHE PREPROCESS ===================
# ======================================================

def apply_clahe(image):
    try:
        image = image.numpy().astype(np.uint8)
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        merged = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
        return enhanced
    except Exception as e:
        print(f"CLAHE error: {e}")
        return image

def preprocess_batch_clahe(images, labels):
    def clahe_fn_single(img):
        return apply_clahe(img)
    images_processed = tf.map_fn(
        lambda x: tf.py_function(clahe_fn_single, [x], tf.uint8),
        images,
        fn_output_signature=tf.uint8
    )
    images_processed.set_shape([None, IMG_SIZE[0], IMG_SIZE[1], 3])
    images_processed = tf.cast(images_processed, tf.float32)
    return images_processed, labels

# ---------------- LOAD DATA ----------------
print("Loading mini datasets...")

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TRAIN_DIR,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TRAIN_DIR,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = train_ds.class_names
print(f"Classes: {class_names}")

# ---------------- AUGMENTATION ----------------
augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
])

def augment_batch(x, y):
    return augmentation(x, training=True), y

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.map(preprocess_batch_clahe, num_parallel_calls=AUTOTUNE)
val_ds   = val_ds.map(preprocess_batch_clahe,   num_parallel_calls=AUTOTUNE)
test_ds  = test_ds.map(preprocess_batch_clahe,  num_parallel_calls=AUTOTUNE)
train_ds = train_ds.map(augment_batch, num_parallel_calls=AUTOTUNE)

train_ds = train_ds.cache().shuffle(500).prefetch(AUTOTUNE)
val_ds   = val_ds.cache().prefetch(AUTOTUNE)
test_ds  = test_ds.cache().prefetch(AUTOTUNE)

# ======================================================
# ================= ATTENTION MODULE ===================
# ======================================================

def se_block(input_tensor, reduction=16):
    filters = input_tensor.shape[-1]
    se = layers.GlobalAveragePooling2D()(input_tensor)
    se = layers.Reshape((1, 1, filters))(se)
    se = layers.Dense(filters // reduction, activation='relu',
                      kernel_initializer='he_normal', use_bias=False)(se)
    se = layers.Dense(filters, activation='sigmoid',
                      kernel_initializer='he_normal', use_bias=False)(se)
    return layers.multiply([input_tensor, se])

# ---------------- MODEL ----------------
def build_model():
    inputs = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    base_model = tf.keras.applications.EfficientNetV2B0(
        include_top=False,
        weights="imagenet",
        input_tensor=inputs,
        include_preprocessing=True
    )
    base_model.trainable = False
    x = base_model.output
    x = se_block(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation="relu",
                     kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = layers.Dropout(0.5)(x)
    output = layers.Dense(1, activation="sigmoid")(x)
    model = models.Model(inputs=inputs, outputs=output,
                         name="Breast_EfficientNet_CLAHE_Mini")
    return model, base_model

model, base_model = build_model()
model.summary()

# ---------------- COMPILE ----------------
try:
    optimizer = tf.keras.optimizers.legacy.Adam()
except:
    optimizer = "adam"

model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1),
    metrics=["accuracy"]
)

# ---------------- CALLBACKS ----------------
early_stop = callbacks.EarlyStopping(
    monitor="val_accuracy", patience=3,
    restore_best_weights=True, verbose=1
)
reduce_lr = callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1
)
checkpoint = callbacks.ModelCheckpoint(
    "breast_cancer_efficientnet_mini.keras",
    monitor="val_accuracy", save_best_only=True, mode="max", verbose=1
)

# ---------------- PHASE 1: TRAINING ----------------
print("\n=== Phase 1: Training Head (Frozen Base) ===\n")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[early_stop, reduce_lr, checkpoint]
)

# ---------------- PHASE 2: FINE TUNING ----------------
print("\n=== Phase 2: Fine Tuning (Unfrozen Top Layers) ===\n")
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False

try:
    optimizer_fine = tf.keras.optimizers.legacy.Adam(1e-5)
except:
    optimizer_fine = tf.keras.optimizers.Adam(1e-5)

model.compile(
    optimizer=optimizer_fine,
    loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1),
    metrics=["accuracy"]
)

history_fine = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS + FINE_TUNE_EPOCHS,
    initial_epoch=history.epoch[-1] + 1 if history.epoch else 0,
    callbacks=[early_stop, reduce_lr, checkpoint]
)

model.save("breast_cancer_efficientnet_mini.keras")
print("\nModel saved to breast_cancer_efficientnet_mini.keras")

# ================= TRAINING GRAPHS =================
acc     = history.history['accuracy']     + history_fine.history['accuracy']
val_acc = history.history['val_accuracy'] + history_fine.history['val_accuracy']
loss    = history.history['loss']         + history_fine.history['loss']
val_loss= history.history['val_loss']     + history_fine.history['val_loss']

ft_start = len(history.history['loss']) - 1

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(acc,     label='Training Accuracy')
axes[0].plot(val_acc, label='Validation Accuracy')
axes[0].axvline(ft_start, color='orange', linestyle='--', label='Fine Tuning Start')
axes[0].legend(loc='lower right')
axes[0].set_title('Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')

axes[1].plot(loss,     label='Training Loss')
axes[1].plot(val_loss, label='Validation Loss')
axes[1].axvline(ft_start, color='orange', linestyle='--', label='Fine Tuning Start')
axes[1].legend(loc='upper right')
axes[1].set_title('Loss')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')

plt.tight_layout()
plt.savefig("breast_training_graphs.png", dpi=150)
print("Training graphs saved to breast_training_graphs.png")
plt.show()

# ================= EVALUATION & CONFUSION MATRIX =================
print("\n=== Evaluation on Test Set ===\n")
y_true = []
y_pred_probs = []

for images, labels in test_ds:
    y_true.extend(labels.numpy())
    probs = model.predict(images, verbose=0)
    y_pred_probs.extend(probs.flatten())

y_true       = np.array(y_true)
y_pred_probs = np.array(y_pred_probs)
y_pred       = (y_pred_probs > 0.5).astype(int)

print(classification_report(y_true, y_pred, target_names=class_names))

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
fig2, ax2 = plt.subplots(figsize=(6, 5))
disp.plot(cmap="Blues", ax=ax2)
ax2.set_title("Confusion Matrix")
plt.tight_layout()
plt.savefig("breast_confusion_matrix.png", dpi=150)
print("Confusion matrix saved to breast_confusion_matrix.png")
plt.show()

print("\nAll done!")
