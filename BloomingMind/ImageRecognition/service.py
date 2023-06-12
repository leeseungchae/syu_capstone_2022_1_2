import os
import urllib.request
import warnings

import pandas as pd
import torch
from PIL import Image
from torchvision import models

from BloomingMind.settings.base import ROOT_DIR
from ImageRecognition import model_url
from ImageRecognition.src.utils import create_transforms

# Disable all warnings
warnings.filterwarnings("ignore")


# Filter out the NNPACK warning


class ImageRecognizer:
    """
    이미지 를 인식 하여 꽃의 이름을 알려 준다.
        Attributes:
            image_path(str): 이미지 경로
    """

    def __init__(self, image_path) -> None:
        self.label_dict = None
        self.model = None
        self.image_path = image_path
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.df = pd.read_csv(os.path.join(ROOT_DIR, "data", "labels.csv"))
        self.num_classes = len(self.df["labels"])
        self.model_load()

    def model_load(self):
        """
        사전에 학습 되어 있는 모델을 불러 오고 업데이트 하여 모델을 불러오는 함수

        """
        self.model = models.mobilenet_v2(pretrained=False)
        self.model.classifier[1] = torch.nn.Linear(1280, self.num_classes)
        model_path = os.path.join(ROOT_DIR, "ImageRecognition", "model", "best.pt")
        if os.path.exists(model_path):
            pass
        else:
            os.makedirs(
                os.path.join(ROOT_DIR, "ImageRecognition", "model"), exist_ok=True
            )
            urllib.request.urlretrieve(model_url, model_path)

        self.model.load_state_dict(
            torch.load(
                os.path.join(model_path),
                map_location=torch.device("cpu"),
            )
        )
        self.model = self.model.to(self.device)

    def search_key(self, index: int):
        """
        가장 확률이 높은 인덱스 를 가지고 key 값을 찾는 함수
            Args:
                index(str): 가장 확률이 높은 인덱스 번호

            Returns:
                key(str) : 꽃의 이름
        """
        names = self.df["names"]
        labels = self.df["labels"]
        self.label_dict = {name: label for name, label in zip(names, labels)}
        found_key = None
        for key, value in self.label_dict.items():
            if value == index:
                found_key = key
                return found_key

    def inference(self) -> str:
        """
        입력된 이미지 를 모델을 통하여 추론 하는 함수
            Returns:
                True : key = 이미지 이름
                False : 이미지 분석 실패
        """

        transform_dict = create_transforms(inference=True)
        data_transform = transform_dict["inference"]

        result_dict = {}

        with torch.no_grad():
            img = Image.open(self.image_path).convert("RGB")
            img = data_transform(img)
            img = img.unsqueeze(0)
            img = img.to(self.device)
            outputs = self.model(img)
            topk_values, topk_indices = torch.topk(outputs, k=5)
            probabilities_percentage = topk_values * 10
            for prob, idx in zip(probabilities_percentage[0], topk_indices[0]):
                key = self.search_key(idx)
                value = round(prob.item(), 2)
                result_dict[key] = value
        return result_dict
