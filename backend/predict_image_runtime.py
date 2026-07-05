import argparse
import json
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
import cv2


IMG_SIZE = (224, 224)
LUNG_CLASSES = [
    "Adenocarcinoma",
    "Large Cell Carcinoma",
    "Normal",
    "Squamous Cell Carcinoma",
    "Other",
]


def apply_clahe(image_array):
    """
    Applies CLAHE to a single image array (uint8, HxWx3 RGB).
    Converts to LAB, applies CLAHE to the L-channel, converts back to RGB.
    Returns uint8 array - matches exactly what training scripts do.
    """
    try:
        if image_array.dtype != np.uint8:
            image_array = image_array.astype(np.uint8)
        lab = cv2.cvtColor(image_array, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        merged = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
        return enhanced.astype(np.uint8)
    except Exception:
        # Fallback: return original if CLAHE fails
        return image_array.astype(np.uint8)


def load_image(path):
    image = Image.open(path)
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)
    return image


def preprocess_for_model(image):
    """
    Convert PIL image -> CLAHE-enhanced float32 array in [0, 255] range.

    IMPORTANT: All three models (brain, lung, breast) were trained with:
      1. CLAHE applied on uint8 images
      2. EfficientNetV2B0 built with include_preprocessing=True

    EfficientNetV2B0's built-in preprocessing layer expects input in [0, 255]
    and internally rescales / normalises to the range it needs.
    Dividing by 255 here would break the model's internal preprocessing and
    produce incorrect (garbage) predictions.
    """
    arr = np.array(image).astype(np.uint8)   # HxWx3, uint8 [0, 255]
    arr = apply_clahe(arr)                   # CLAHE enhancement (matches training)
    arr = arr.astype(np.float32)             # cast to float32, keep [0, 255] range
    arr = np.expand_dims(arr, axis=0)        # (1, 224, 224, 3)
    return arr


def predict_brain(model, image):
    arr = preprocess_for_model(image)
    pred = float(model.predict(arr, verbose=0)[0][0])
    if pred >= 0.5:
        return {"prediction": "Cancer", "confidence": round(pred * 100, 2)}
    return {"prediction": "No Cancer", "confidence": round((1 - pred) * 100, 2)}


def predict_lung(model, image):
    arr = preprocess_for_model(image)
    preds = model.predict(arr, verbose=0)[0]
    idx = int(np.argmax(preds))
    return {"prediction": LUNG_CLASSES[idx], "confidence": round(float(preds[idx] * 100), 2)}


def predict_breast(model, image):
    arr = preprocess_for_model(image)
    pred = float(model.predict(arr, verbose=0)[0][0])
    if pred >= 0.5:
        return {"prediction": "Cancer", "confidence": round(pred * 100, 2)}
    return {"prediction": "No Cancer", "confidence": round((1 - pred) * 100, 2)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["brain", "lung", "breast"], required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--image-path", required=True)
    args = parser.parse_args()

    model = tf.keras.models.load_model(args.model_path, compile=False, safe_mode=False)
    image = load_image(args.image_path)

    if args.task == "brain":
        result = predict_brain(model, image)
    elif args.task == "lung":
        result = predict_lung(model, image)
    else:
        result = predict_breast(model, image)

    print(json.dumps(result))


if __name__ == "__main__":
    main()
