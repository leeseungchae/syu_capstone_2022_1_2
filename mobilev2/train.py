# Standard library
import copy
import glob
import multiprocessing
import os
import time

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from data_load import make_datasets
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm.notebook import tqdm

# Number of classes in the dataset

feature_extract = True


def train_model(
    model,
    dataloaders,
    criterion,
    optimizer,
    device,
    num_epochs=25,
    save_interval=5,
    patience=5,
):
    since = time.time()

    history = {"accuracy": [], "val_accuracy": [], "loss": [], "val_loss": []}

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    early_stopping_counter = 0  # Counter to track early stopping
    early_stopping_flag = False  # Flag to indicate early stopping

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.float() / len(dataloaders[phase].dataset)

            print("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))

            if phase == "val":
                history["val_accuracy"].append(epoch_acc.item())
                history["val_loss"].append(epoch_loss)

                # Early stopping check
                if epoch_acc <= best_acc:
                    early_stopping_counter += 1
                    if early_stopping_counter >= patience:
                        early_stopping_flag = True
            else:
                history["accuracy"].append(epoch_acc.item())
                history["loss"].append(epoch_loss)

            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        if early_stopping_flag:
            print(
                "Early stopping triggered! No improvement in validation accuracy for {} epochs.".format(
                    patience
                )
            )
            break

        if epoch % save_interval == 0:
            torch.save(model.state_dict(), f"./model/model_epoch_{epoch}.pt")
            print("Saved model at epoch", epoch)

    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    print("Best val Acc: {:4f}".format(best_acc))

    model.load_state_dict(best_model_wts)
    return model, history


if __name__ == "__main__":
    train_dir = "./data/train"
    label_csv = "./data/labels.csv"

    df = pd.read_csv(label_csv)
    image_datasets = make_datasets(train_dir, df)

    # config
    batch_size = 32
    num_epochs = 50
    num_workers = multiprocessing.cpu_count()
    num_classes = len(df["labels"])

    # Create training and validation dataloaders
    dataloaders_dict = {
        x: DataLoader(
            image_datasets[x],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
        for x in ["train", "val"]
    }

    model_ft = models.mobilenet_v2(pretrained=True)
    model_ft.classifier[1] = nn.Linear(1280, num_classes)
    # mac
    device = torch.device("mps") if torch.backends.mps.is_available() else "cpu"
    print(device)
    # mac 제외
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Send the model to GPU
    model_ft = model_ft.to(device)

    params_to_update = model_ft.parameters()
    print("Params to learn:")

    # Observe that all parameters are being optimizedss
    optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)

    criterion = nn.CrossEntropyLoss()

    # Train and evaluate
    model_ft, hist = train_model(
        model_ft,
        dataloaders_dict,
        criterion,
        optimizer_ft,
        num_epochs=num_epochs,
        device=device,
    )

    import os

    # Set the directory path where the model files are saved
    os.makedirs("./model", exist_ok=True)
    directory = "./model/"

    # Get a list of all the model files in the directory
    model_files = [file for file in os.listdir(directory) if file.endswith(".pt")]

    # Sort the model files based on the epoch number
    sorted_files = sorted(model_files, key=lambda x: int(x.split("_")[2].split(".")[0]))

    # Get the highest epoch file
    highest_epoch_file = sorted_files[-1]

    # Rename the highest epoch file to "best.pt"
    best_file_path = os.path.join(directory, "best.pt")
    highest_epoch_file_path = os.path.join(directory, highest_epoch_file)
    os.rename(highest_epoch_file_path, best_file_path)

    # Remove the other model files
    for file in sorted_files[:-1]:
        file_path = os.path.join(directory, file)
        os.remove(file_path)
