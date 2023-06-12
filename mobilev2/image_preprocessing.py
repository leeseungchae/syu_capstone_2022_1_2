import os
import random
import shutil
from math import floor

import pandas as pd

from mobilev2.utils import file_download

# Set the path to the original folder containing image folders
original_folder = "./data/original"
train_folder = "./data/train"
test_folder = "./data/test"

# if not os.path.exists(env_path):
#     os.makedirs(os.path.join(ROOT_DIR, 'Core', 'env'), exist_ok=True)
#     write_random_secret_key()
if not os.path.exists(original_folder):
    os.makedirs(os.path.join(original_folder))
    file_download()
else:
    subfolders = [f.name for f in os.scandir(original_folder) if f.is_dir()]
    if not subfolders:
        file_download()
folder_list = os.listdir(original_folder)


names = []
labels = []

for idx, name in enumerate(folder_list):
    names.append(name)
    labels.append(idx)

df = pd.DataFrame({"names": names, "labels": labels})

os.makedirs("./data", exist_ok=True)
df.to_csv("./data/labels.csv", index=False)


subfolders = [f.path for f in os.scandir(original_folder) if f.is_dir()]
min_count = float("inf")

for folder in subfolders:
    file_count = len(os.listdir(folder))
    if file_count < min_count:
        min_count = file_count
# Create the train folder if it doesn't exist
os.makedirs(train_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)


# Calculate the number of files for training and testing
total_count = min_count
train_count = floor(0.8 * min_count)
test_count = min_count - train_count

# Iterate over each image folder in the original folder
for folder in os.listdir(original_folder):
    folder_path = os.path.join(original_folder, folder)

    # Check if the current item is a directory
    if os.path.isdir(folder_path):
        # Get a list of image file names in the current folder
        images = os.listdir(folder_path)

        # Shuffle the list of images
        random.shuffle(images)

        # Calculate the number of images for train and test based on the split percentage
        # Split the images into train and test lists
        train_images = images[:train_count]
        test_images = images[:test_count]

        # Copy and rename the train images
        for i, image in enumerate(train_images):
            image_path = os.path.join(folder_path, image)
            new_image_name = f"{folder}.{i + 1}.jpg"  # Modify the extension if your image files have a different format
            new_image_path = os.path.join(train_folder, new_image_name)
            shutil.copy(image_path, new_image_path)

        # Copy and rename the test images
        for i, image in enumerate(test_images):
            image_path = os.path.join(folder_path, image)
            new_image_name = f"{folder}.{i + 1}.jpg"  # Modify the extension if your image files have a different format
            new_image_path = os.path.join(test_folder, new_image_name)
            shutil.copy(image_path, new_image_path)

        print("Split and copied images from folder:", folder)

    print(
        "Images have been split and copied to the train and test folders with the specified names."
    )
