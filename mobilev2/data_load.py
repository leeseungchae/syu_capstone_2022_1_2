import glob
import os

import torch
from ImageRecognition.src.utils import create_transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


class FlowerDataset(Dataset):
    def __init__(self, file_list, df, transform=None):
        self.file_list = file_list
        name_list = df["names"].tolist()
        label_list = df["labels"].tolist()
        self.label_dict = {name: label for name, label in zip(name_list, label_list)}
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        self.img_path = self.file_list[idx]
        from PIL import ImageFile

        ImageFile.LOAD_TRUNCATED_IMAGES = True
        image = Image.open(self.img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        label = self.extract_class_from()

        return image, label

    def extract_class_from(self):
        file = self.img_path.split("/")[-1]
        name = file.split(".")[0]
        label = self.label_dict[name]

        return label


def make_datasets(train_dir, df):
    all_train_files = glob.glob(os.path.join(train_dir, "*.jpg"))

    train_list, val_list = train_test_split(all_train_files, random_state=42)
    data_transforms = create_transforms(train=True, val=True)

    image_datasets = {
        "train": FlowerDataset(train_list, df, transform=data_transforms["train"]),
        "val": FlowerDataset(val_list, df, transform=data_transforms["val"]),
    }
    return image_datasets
