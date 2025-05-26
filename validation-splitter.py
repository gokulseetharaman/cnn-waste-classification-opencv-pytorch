import os
import shutil
import random

# Source folder (after extracting the Kaggle zip)
SOURCE_DIR = "garbage_classification"

# Destination base folders
TRAIN_DIR = "dataset/train"
VAL_DIR = "dataset/val"

# Validation split ratio
VAL_RATIO = 0.2

# Set random seed for reproducibility
random.seed(42)

# Get list of class names (subdirectories in source)
classes = [cls for cls in os.listdir(SOURCE_DIR) if os.path.isdir(os.path.join(SOURCE_DIR, cls))]

# Create train/val folders
for split_dir in [TRAIN_DIR, VAL_DIR]:
    os.makedirs(split_dir, exist_ok=True)
    for cls in classes:
        os.makedirs(os.path.join(split_dir, cls), exist_ok=True)

# Split each class folder
for cls in classes:
    cls_path = os.path.join(SOURCE_DIR, cls)
    images = [img for img in os.listdir(cls_path) if os.path.isfile(os.path.join(cls_path, img))]

    random.shuffle(images)
    val_count = int(len(images) * VAL_RATIO)
    val_images = images[:val_count]
    train_images = images[val_count:]

    # Copy to train
    for img in train_images:
        src = os.path.join(cls_path, img)
        dst = os.path.join(TRAIN_DIR, cls, img)
        shutil.copyfile(src, dst)

    # Copy to val
    for img in val_images:
        src = os.path.join(cls_path, img)
        dst = os.path.join(VAL_DIR, cls, img)
        shutil.copyfile(src, dst)

print("✅ Dataset has been successfully split into train and validation sets.")
