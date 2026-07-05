import os
import shutil
import random
import glob

# -------- CONFIG --------
SOURCE_DIR = "IDC_regular_ps50_idx5"
DEST_DIR = "dataset_split"
TRAIN_RATIO = 0.8
SEED = 42

random.seed(SEED)

# Define classes based on folder structure
CLASSES = ["negative_IDC", "positive_IDC"]

# Create train and test folders
for split in ["train", "test"]:
    for cls in CLASSES:
        os.makedirs(os.path.join(DEST_DIR, split, cls), exist_ok=True)

print("Starting dataset split...")

# Loop through each class folder
for class_name in CLASSES:
    class_path = os.path.join(SOURCE_DIR, class_name)

    if not os.path.isdir(class_path):
        print(f"Warning: {class_path} not found!")
        continue

    # Get all images in the class folder (recursively or flat? usually flat for this structure)
    # The structure seems to be breast/IDC_regular_ps50_idx5/negative_IDC/<images> based on list_dir
    images = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    random.shuffle(images)

    split_index = int(len(images) * TRAIN_RATIO)
    train_images = images[:split_index]
    test_images = images[split_index:]

    # Create class folders in destination
    train_class_dir = os.path.join(DEST_DIR, "train", class_name)
    test_class_dir = os.path.join(DEST_DIR, "test", class_name)

    # Copy images
    print(f"Processing {class_name}...")
    for img in train_images:
        shutil.copy(
            os.path.join(class_path, img),
            os.path.join(train_class_dir, img)
        )

    for img in test_images:
        shutil.copy(
            os.path.join(class_path, img),
            os.path.join(test_class_dir, img)
        )

    print(f"  -> {len(train_images)} train images")
    print(f"  -> {len(test_images)} test images")

print("\n✅ Dataset split completed successfully!")
