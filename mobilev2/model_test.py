import glob
import os
import time
from collections import Counter

import pandas as pd
import torch
from ImageRecognition.src.utils import create_transforms
from PIL import Image, ImageFile
from torchvision import models

ImageFile.LOAD_TRUNCATED_IMAGES = True

save_path = "./model/best.pt"
label_csv = "./data/labels.csv"
test_dir = "./data/test"

df = pd.read_csv(label_csv)
# Create an instance of the MobileNetV2 model
model = models.mobilenet_v2(pretrained=False)
# Modify the classifier layer

num_classes = len(
    df["labels"]
)  # Replace with the correct number of output classes in the saved model
model.classifier[1] = torch.nn.Linear(1280, num_classes)

# Load the saved model weights
model.load_state_dict(torch.load(save_path))

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# mac
device = torch.device("mps") if torch.backends.mps.is_available() else "cpu"
model = model.to(device)
model.eval()

# Set the model to evaluation mode

test_list = glob.glob(os.path.join(test_dir, "*.jpg"))
transform_dict = create_transforms(inference=True)
test_data_transform = transform_dict["inference"]
name_list = df["names"]
label_list = df["labels"]
label_dict = {name: label for name, label in zip(name_list, label_list)}


def extract_class_from(path):
    file = path.split("/")[-1]
    name = file.split(".")[0]
    label = label_dict[name]

    return label


preds = []
targets = []
matches = []
probabilitys = []
# test_dir = '/home/steve/workspace/syu_capstone/BloomingMind/data/image'
# test_list = glob.glob(os.path.join(test_dir, '*.jpg'))

start_time = time.time()
with torch.no_grad():
    for test_path in test_list:
        img = Image.open(test_path).convert("RGB")
        img = test_data_transform(img)
        img = img.unsqueeze(0)
        img = img.to(device)

        outputs = model(img)
        max_index = torch.argmax(outputs)
        preds.append(max_index.item())
        target = extract_class_from(test_path)
        targets.append(target)
        # print(outputs[0,max_index].item())
        probabilitys.append(outputs[0, max_index].item())

        if max_index == target:
            matches.append("o")
        else:
            matches.append("x")
print(time.time() - start_time)


output = pd.DataFrame(
    {
        "targets": targets,
        "preds": preds,
        "matches": matches,
        "probabilitys": probabilitys,
    }
)
is_correct = Counter(item for item in matches if "o" in item)

print("정답률", is_correct["o"] / len(test_list) * 100, "%")

# output.sort_values(by='id', inplace=True)
# output.reset_index(drop=True, inplace=True)
os.makedirs("./data/result", exist_ok=True)
output.to_csv("./data/submission.csv", index=False)
