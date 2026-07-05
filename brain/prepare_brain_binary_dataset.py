import os
import shutil

SOURCE_DIR = "dataset"
DEST_DIR = "brain_binary"

CANCER_CLASSES = ["glioma", "meningioma", "pituitary"]
NO_CANCER_CLASS = "notumor"

def make_dirs():
    for split in ["Training", "Testing"]:
        for cls in ["Cancer", "No_Cancer"]:
            os.makedirs(os.path.join(DEST_DIR, split, cls), exist_ok=True)

def copy_class(class_name, target_label):
    for split in ["Training", "Testing"]:
        src = os.path.join(SOURCE_DIR, split, class_name)
        images = [f for f in os.listdir(src) if f.lower().endswith((".jpg", ".png", ".jpeg"))]

        for img in images:
            shutil.copy(
                os.path.join(src, img),
                os.path.join(DEST_DIR, split, target_label, img)
            )

make_dirs()

# Copy cancer images
for cls in CANCER_CLASSES:
    copy_class(cls, "Cancer")

# Copy no-cancer images
copy_class(NO_CANCER_CLASS, "No_Cancer")

print("✅ Brain dataset converted to binary while preserving Training/Testing split")
