import tkinter as tk
from tkinter import filedialog

from PIL import Image

from mobilev2.utils import create_transforms

# Tkinter 창 초기화
root = tk.Tk()
root.withdraw()

# 이미지 파일 선택 대화상자 열기
file_path = filedialog.askopenfilename(
    title="Select Image File",
    filetypes=[("Image Files", ("*.jpg", "*.jpeg", "*.png", "*.gif"))],
)

# 선택한 파일 경로 출력
print("Selected File:", file_path)

# 선택한 이미지 열기
image = Image.open(file_path)

# Tkinter 창 종료
root.destroy()


import glob
import time
from collections import Counter

import pandas as pd
import torch
from PIL import Image, ImageFile
from torchvision import models

ImageFile.LOAD_TRUNCATED_IMAGES = True

label_csv = "./data/labels.csv"
df = pd.read_csv(label_csv)
save_path = "./model/best.pt"

# Create an instance of the MobileNetV2 model
model = models.mobilenet_v2(pretrained=False)
# Modify the classifier layer
num_classes = len(
    df["labels"]
)  # Replace with the correct number of output classes in the saved model
model.classifier[1] = torch.nn.Linear(1280, num_classes)

# Load the saved model weights
model.load_state_dict(torch.load(save_path))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = model.to(device)
model.eval()

# Set the model to evaluation mode


transform_dict = create_transforms(inference=True)
test_data_transform = transform_dict["inference"]
name_list = df["names"]
label_list = df["labels"]
label_dict = {name: label for name, label in zip(name_list, label_list)}


def search_key(index):
    found_key = None
    for key, value in label_dict.items():
        if value == index:
            found_key = key
            return found_key


start_time = time.time()
with torch.no_grad():
    img = image.convert("RGB")
    img = test_data_transform(img)
    img = img.unsqueeze(0)
    img = img.to(device)

    outputs = model(img)
    max_index = torch.argmax(outputs)
    key = search_key(max_index.item())

    probability = outputs[0, max_index].item()

    print(f"이꽃의 이름은 {(round(probability * 10))}%의 확률로 {key}일 것 같습니다")
