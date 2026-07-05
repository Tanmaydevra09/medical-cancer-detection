import os
import shutil
import random

# -------- CONFIG --------
SOURCE_DIR = "Lung Cancer Dataset"
DEST_DIR = "dataset_split"
TRAIN_RATIO = 0.8
SEED = 42

random.seed(SEED)

# Create train and test folders
for split in ["train", "test"]:
    os.makedirs(os.path.join(DEST_DIR, split), exist_ok=True)

# Loop through each class folder
for class_name in os.listdir(SOURCE_DIR):
    class_path = os.path.join(SOURCE_DIR, class_name)

    if not os.path.isdir(class_path):
        continue

    images = os.listdir(class_path)
    random.shuffle(images)

    split_index = int(len(images) * TRAIN_RATIO)
    train_images = images[:split_index]
    test_images = images[split_index:]

    # Create class folders
    train_class_dir = os.path.join(DEST_DIR, "train", class_name)
    test_class_dir = os.path.join(DEST_DIR, "test", class_name)

    os.makedirs(train_class_dir, exist_ok=True)
    os.makedirs(test_class_dir, exist_ok=True)

    # Copy images
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

    print(f"{class_name}: {len(train_images)} train, {len(test_images)} test")

print("✅ Dataset split completed!")
