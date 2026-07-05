import tensorflow as tf
from tensorflow.keras import layers, models, callbacks # type: ignore
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


# ---------------- CONFIG ----------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
FINE_TUNE_EPOCHS = 5
SEED = 42

print(f"TensorFlow Version: {tf.__version__}")

# ======================================================
# ================= CLAHE PREPROCESS ===================
# ======================================================

def apply_clahe(image):
    """
    Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) to a single image.
    Converts to LAB color space, applies CLAHE to L-channel, and converts back to RGB.
    """
    try:
        image = image.numpy().astype(np.uint8)
        
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        
        merged = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
        
        return enhanced
    except Exception as e:
        print(f"Error in apply_clahe: {e}")
        return image

def preprocess_with_clahe(image, label):
    
    def clahe_fn(img):
        return apply_clahe(img)

    
    
    # We need to wrap the processed image back specific dtype
    image = tf.py_function(clahe_fn, [image], tf.uint8)
    
    # Ensure shape is preserved (py_function loses shape info)
    image.set_shape([IMG_SIZE[0], IMG_SIZE[1], 3])
    
    # Cast to float32 for model, but keep range [0, 255]
    image = tf.cast(image, tf.float32)
    
    return image, label

# ---------------- LOAD DATA ----------------
# Using 'Training' folder for Train/Val split
print("Loading datasets...")

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "brain_binary/Training",
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "brain_binary/Training",
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)


# Test set from 'Testing' folder
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "brain_binary/Testing",
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


def preprocess_batch_clahe(images, labels):
    """
    Applies CLAHE to a batch of images.
    """
    def clahe_fn_single(img):
        return apply_clahe(img)

    # Use map_fn to apply to each image in the batch
    # fn_output_signature must match the output of clahe_fn_single (tensor of uint8)
    images_processed = tf.map_fn(
        lambda x: tf.py_function(clahe_fn_single, [x], tf.uint8),
        images,
        fn_output_signature=tf.uint8
    )
    
    # Restore shape info for the batch
    images_processed.set_shape([None, IMG_SIZE[0], IMG_SIZE[1], 3])
    
    # Cast to float32 [0, 255]
    images_processed = tf.cast(images_processed, tf.float32)
    
    return images_processed, labels

def augment_batch(x, y):
    return augmentation(x, training=True), y

# Apply CLAHE
train_ds = train_ds.map(preprocess_batch_clahe)
val_ds   = val_ds.map(preprocess_batch_clahe)
test_ds  = test_ds.map(preprocess_batch_clahe)

# Apply Augmentation (Train only)
train_ds = train_ds.map(augment_batch)

# Optimization
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)
val_ds   = val_ds.cache().prefetch(AUTOTUNE)
test_ds  = test_ds.cache().prefetch(AUTOTUNE)

# ======================================================
# ================= ATTENTION MODULE ===================
# ======================================================

def se_block(input_tensor, reduction=16):
    filters = input_tensor.shape[-1]
    
    se = layers.GlobalAveragePooling2D()(input_tensor)
    se = layers.Reshape((1, 1, filters))(se)
    se = layers.Dense(filters // reduction, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = layers.Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
    
    return layers.multiply([input_tensor, se])

# ---------------- MODEL ----------------
def build_model():
    # Input
    inputs = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    
    # Base Model: EfficientNetV2B0
    # include_preprocessing=True handles [0, 255] -> [-1, 1] or similar standard.
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
    x = layers.Dense(256, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = layers.Dropout(0.5)(x)
    output = layers.Dense(1, activation="sigmoid")(x)
    
    model = models.Model(inputs=inputs, outputs=output, name="EfficientNet_CLAHE_Attention")
    return model, base_model

model, base_model = build_model()
model.summary()

# ---------------- COMPILE ----------------
model.compile(
    optimizer="adam",
    loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1),
    metrics=["accuracy"]
)

# ---------------- CALLBACKS ----------------
early_stop = callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=4,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    min_lr=1e-6,
    verbose=1
)

# ---------------- TRAINING ----------------
print("\n=== Phase 1: Training Head (Frozen Base) ===\n")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[early_stop, reduce_lr]
)

# ---------------- FINE TUNING ----------------
print("\n=== Phase 2: Fine Tuning (Unfrozen Top Layers) ===\n")
base_model.trainable = True

# Fine-tune the last 20 layers
for layer in base_model.layers[:-20]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1),
    metrics=["accuracy"]
)

history_fine = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS + FINE_TUNE_EPOCHS,
    initial_epoch=history.epoch[-1] + 1 if history.epoch else 0, # Should rely on len(history.history)
    callbacks=[early_stop, reduce_lr]
)

model.save("brain_binary_efficientnet_clahe.keras")
print("\nModel saved to brain_binary_efficientnet_clahe.keras")

# ================= TRAINING GRAPHS =================
# Combine history
acc = history.history['accuracy'] + history_fine.history['accuracy']
val_acc = history.history['val_accuracy'] + history_fine.history['val_accuracy']
loss = history.history['loss'] + history_fine.history['loss']
val_loss = history.history['val_loss'] + history_fine.history['val_loss']

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.plot([len(history.history['loss'])-1, len(history.history['loss'])-1],
         plt.ylim(), label='Fine Tuning Start')
plt.legend(loc='lower right')
plt.title('Accuracy')

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.plot([len(history.history['loss'])-1, len(history.history['loss'])-1],
         plt.ylim(), label='Fine Tuning Start')
plt.legend(loc='upper right')
plt.title('Loss')
plt.show()

# ================= EVALUATION & CONFUSION MATRIX =================
print("\n=== Evaluation on Test Set ===\n")

y_true = []
y_pred_probs = []

for images, labels in test_ds:
    y_true.extend(labels.numpy())
    probs = model.predict(images, verbose=0)
    y_pred_probs.extend(probs.flatten())

y_true = np.array(y_true)
y_pred_probs = np.array(y_pred_probs)
y_pred = (y_pred_probs > 0.5).astype(int)

class_names_list = ["No Cancer", "Cancer"] # Assuming binary
if len(class_names) == 2:
    class_names_list = class_names

print(classification_report(y_true, y_pred, target_names=class_names_list))

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=class_names_list)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()

# ======================================================
# ===================== GRAD-CAM =======================
# ======================================================
print("\n=== Generating Grad-CAM ===\n")

def get_last_conv_layer_name(model):
    """
    Dynamically find the last 4D (Conv) layer in the base model.
    EfficientNet blocks usually differ in naming.
    """
    for layer in reversed(model.layers):
        if len(layer.output.shape) == 4:
            return layer.name
    return None

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # 'base_model' is a layer in 'model'. Access it.
    base_layer = model.get_layer(name=base_model.name)
    
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [base_layer.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# Test image (using 'img1.jpg' if exists, else first from test set)
img_path = "img1.jpg"

if os.path.exists(img_path):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=IMG_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    # CLAHE for single image
    img_array_clahe = apply_clahe(tf.convert_to_tensor(img_array))
    img_array_input = np.expand_dims(img_array_clahe, axis=0) # (1, 224, 224, 3)
    
    # Find layer
    last_conv_layer = get_last_conv_layer_name(base_model)
    print(f"Grad-CAM Target Layer: {last_conv_layer}")
    
    if last_conv_layer:
        try:
            heatmap = make_gradcam_heatmap(img_array_input, model, last_conv_layer)

            img_cv = cv2.imread(img_path)
            img_cv = cv2.resize(img_cv, IMG_SIZE)
            
            # Apply CLAHE to display image too for consistency? Or show original?
            # User showed Original in subplot 1.
            
            heatmap = cv2.resize(heatmap, IMG_SIZE)
            heatmap = np.uint8(255 * heatmap)
            heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            
            overlay = cv2.addWeighted(img_cv, 0.6, heatmap_color, 0.4, 0)

            plt.figure(figsize=(12,4))
            
            plt.subplot(1,3,1)
            plt.imshow(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
            plt.title("Original"); plt.axis("off")

            plt.subplot(1,3,2)
            plt.imshow(heatmap, cmap="jet")
            plt.title("Grad-CAM Heatmap"); plt.axis("off")

            plt.subplot(1,3,3)
            plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
            plt.title("Tumor Localization"); plt.axis("off")

            plt.show()
        except Exception as e:
            print(f"Grad-CAM error: {e}")
else:
    print(f"Image {img_path} not found.")
