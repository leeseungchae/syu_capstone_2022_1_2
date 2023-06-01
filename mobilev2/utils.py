import os.path
import traceback
import urllib.request

from torchvision import transforms


def create_transforms(train=False, val=False, inference=False):
    """
    transform을 만드는 함수
    Args:
        train(str):
        val(str):
        inference(str):

    Returns:
        transform_dict : transfrom 정보들
    """
    transform_dict = {}

    if train:
        transform_dict["train"] = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
    if val:
        transform_dict["val"] = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
    if inference:
        transform_dict["inference"] = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    if inference:
        transform_dict["test"] = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    return transform_dict


def file_download():
    import shutil
    from mobilev2 import image_path
    import zipfile
    save_path = os.path.join('data', 'original', 'flower.zip')
    urllib.request.urlretrieve(image_path, save_path)
    target_folder = os.path.join(os.getcwd(), 'data', 'original')
    with zipfile.ZipFile(save_path, 'r') as zip_ref:
        zip_ref.extractall(target_folder)

    for folder_name in os.listdir(os.path.join(target_folder, 'flowers')):
        folder_path = os.path.join(os.path.join(target_folder, 'flowers'), folder_name)
        if os.path.isdir(target_folder):
            shutil.move(folder_path, target_folder)

    shutil.rmtree(os.path.join(target_folder, 'flowers'))
