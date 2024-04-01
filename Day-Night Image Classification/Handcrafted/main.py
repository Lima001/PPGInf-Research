import os
import torch
import argparse
import numpy as np
from PIL import Image
import torchvision
from torch.utils.data import DataLoader
from torcheval.metrics.functional import multiclass_confusion_matrix
from albumentations import *
from albumentations.pytorch import ToTensorV2
import method1
import method2
import method3

BATCH_SIZE = 16

class Transform():

    def __init__(self,transform):
        self.transform=transform

    def __call__(self,image):
        return self.transform(image=image)["image"]

    @staticmethod
    def open_img(img_path):
        img = Image.open(img_path)
        return np.array(img)

    @staticmethod
    def open_hsv_img(img_path):
        img = Image.open(img_path).convert("HSV")
        return np.array(img)

if __name__ == "__main__":

    torch.set_num_threads(1)

    parser = argparse.ArgumentParser()
    parser.add_argument("--device")
    parser.add_argument("--data_dir")
    parser.add_argument("--method")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    data_transform = Transform(Compose([
        Resize(224,224),
        #Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ]))

    if args.method == "method3":
        loader = Transform.open_hsv_img
    else:
        loader = Transform.open_img

    day_dataset = torchvision.datasets.ImageFolder(os.path.join(args.data_dir, "train"),
                                                          data_transform,
                                                          loader=loader)
    day_dataset.class_to_idx = {"day":0, "night":1}
    idx = day_dataset.targets == 0
    day_dataset.targets = day_dataset.targets[idx]
    day_dataset_size = len(day_dataset)

    day_dataloader = torch.utils.data.DataLoader(day_dataset, batch_size=BATCH_SIZE, shuffle=True)


    image_datasets = {
        x: torchvision.datasets.ImageFolder(os.path.join(args.data_dir, x),
                                            data_transform,
                                            loader=loader)
        for x in ['train', 'val']
    }

    dataloaders = {
        x: torch.utils.data.DataLoader( image_datasets[x],
                                        batch_size=BATCH_SIZE,
                                        shuffle=True,
                                        num_workers=0,
                                        pin_memory=False)
        for x in ['train', 'val']
    }

    dataset_sizes = {
            x: len(image_datasets[x])
            for x in ['train', 'val']
    }

    class_names = image_datasets['train'].classes

    if args.method == "method1":
        threshold = method1.get_avg_brightness(device, day_dataloader, day_dataset_size)
        print(threshold)
        method1.classify_dataset(device, dataloaders, class_names, dataset_sizes, threshold)

    elif args.method == "method2":
        threshold_h, threshold_v = method2.get_avg_thresholds(device, day_dataloader, day_dataset_size)
        print(threshold_h, threshold_v)
        method2.classify_dataset(device, dataloaders, class_names, dataset_sizes, threshold_h, threshold_v)

    elif args.method == "method3":
        method3.classify_dataset(device, dataloaders, class_names, dataset_sizes)
