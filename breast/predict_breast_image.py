import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

# ---------------- CONFIG ----------------
MODEL_PATH = "breast_cancer_efficientnet_clahe.keras"
IMG_PATH = "img1.png"
IMG_SIZE = (224, 224)

# ---------------- LOAD MODEL ----------------
print(f"Loading model from {MODEL_PATH}...")
try:
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        print("Model loaded successfully.")
    else:
        print(f"Model file not found at {MODEL_PATH}")
        print("Please train the model first by running train_breast_efficientnet_clahe.py")
        exit()
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# ---------------- CLAHE PREPROCESS ----------------
def apply_clahe(image_array):
    """
    Applies CLAHE to a single image array (H, W, 3).
    Expects float input or uint8. Returns float32.
    """
    try:
        # Convert to uint8 if needed
        if image_array.dtype != np.uint8:
            image = image_array.astype(np.uint8)
        else:
            image = image_array
        
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        
        merged = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
        
        return enhanced.astype(np.float32)
    except Exception as e:
        print(f"Error in CLAHE: {e}")
        return image_array

# ---------------- IMAGE PREPROCESSING ----------------
def preprocess_image(path):
    if not os.path.exists(path):
        print(f"Error: Image {path} not found.")
        return None, None
        
    img = tf.keras.preprocessing.image.load_img(path, target_size=IMG_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    
    # Apple CLAHE
    img_clahe = apply_clahe(img_array)
    
    # Expand dims (1, 224, 224, 3)
    # EfficientNetV2 handles [0, 255] range internally
    img_expanded = np.expand_dims(img_clahe, axis=0)
    
    return img_expanded, img_array # Return original array for display? Or CLAHE?

img_tensor, img_original = preprocess_image(IMG_PATH)
if img_tensor is None:
    exit()

# ---------------- PREDICTION ----------------
print("Running prediction...")
print(f"Tensor shape: {img_tensor.shape}, Range: [{np.min(img_tensor)}, {np.max(img_tensor)}]")
preds = model.predict(img_tensor)
score = preds[0][0]

# Sigmoid output: 0.0 -> Class 0 (negative_IDC), 1.0 -> Class 1 (positive_IDC)
if score >= 0.5:
    label = "Positive (IDC)"
    confidence = score * 100
else:
    label = "Negative (No IDC)"
    confidence = (1 - score) * 100

print(f"Prediction: {label} ({confidence:.2f}%)")

# ---------------- GRAD-CAM ----------------
def get_last_conv_layer_name(model):
    # Find last Conv2D from the flattened model layers
    for layer in reversed(model.layers):
        if len(layer.output.shape) == 4 and isinstance(layer, tf.keras.layers.Conv2D):
             return layer.name
    return None

def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    # If using Functional API where inputs are shared, access layers directly
    last_conv_layer = model.get_layer(last_conv_layer_name)
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [last_conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        class_channel = preds[:, 0]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

layer_name = get_last_conv_layer_name(model)

if layer_name:
    print(f"Generating Grad-CAM using layer: {layer_name}")
    try:
        heatmap = make_gradcam_heatmap(img_tensor, model, layer_name)

        # Visualization
        img_cv = cv2.imread(IMG_PATH)
        img_cv = cv2.resize(img_cv, IMG_SIZE)
        
        heatmap_resized = cv2.resize(heatmap, IMG_SIZE)
        heatmap_resized = np.uint8(255 * heatmap_resized)
        heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)

        overlay = cv2.addWeighted(img_cv, 0.6, heatmap_color, 0.4, 0)

        plt.figure(figsize=(10, 4))
        plt.subplot(1, 4, 1)
        plt.imshow(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
        plt.title(f"Input\n{label}\n({confidence:.1f}%)")
        plt.axis('off')

        plt.subplot(1, 4, 2)
        plt.imshow(heatmap, cmap='jet')
        plt.title("Attention Map")
        plt.axis('off')

        plt.subplot(1, 4, 3)
        plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        plt.title("Overlay")
        plt.axis('off')
        
        # Add a histogram of prediction to look "more accurate" / "scientific"
        plt.subplot(1, 4, 4)
        bars = plt.bar(["Negative", "Positive"], [1-score, score], color=['green', 'red'])
        plt.title("Confidence info")
        plt.ylim(0, 1)
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Grad-CAM failed: {e}")
else:
    print("Skipping Grad-CAM - No Convolutional Layer Found.")
