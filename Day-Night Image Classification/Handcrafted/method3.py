# Inspired by:
#   K. H. Park and Y. S. Lee
#   Classification of Daytime and Night Based on Intensity and Chromaticity in RGB Color Image
#   2018 International Conference on Platform Technology and Service (PlatCon) 
#   Jeju, Korea (South), 2018, pp. 1-6, doi: 10.1109/PlatCon.2018.8472764
# classification method
# Method description: Day-Night Classifier based on Intensity in RGB Color Image

# Obs. Chromacity and K-means segementation - presented in the paper - were not used!

import torch
from torcheval.metrics.functional import multiclass_confusion_matrix
from tqdm import tqdm

BASE_NIGHT_DAY = 65

def classify_dataset(device, dataloaders, class_names, dataset_sizes):

    for phase in ['train', 'val']:
        if phase == "val":
            confusion_matrix = torch.zeros((len(class_names), len(class_names))).to(device)

        running_corrects = 0

        with tqdm(dataloaders[phase], unit="batch") as t:

            for inputs, targets in t:
                inputs, targets = inputs.to(device), targets.to(device)
                intensity_inputs = inputs.mean(dim=1, dtype=torch.float)
                night_pixels = (intensity_inputs <= BASE_NIGHT_DAY).int().sum(dim=[1,2])
                day_pixels = (intensity_inputs > BASE_NIGHT_DAY).int().sum(dim=[1,2])
                preds = (night_pixels > day_pixels).int()

                if phase == "val":
                    confusion_matrix += multiclass_confusion_matrix(targets, preds, len(class_names)).to(device)

                running_corrects += torch.sum(preds == targets)

            acc = running_corrects.double() / dataset_sizes[phase]
            print(f'{phase} acc: {acc:.4f}')

            if phase == "val":
                print("Confusion matrix (raw counts)")
                for i in range(len(class_names)):
                    for j in range(len(class_names)):
                        print(int(confusion_matrix[i][j]), end="\t")
                    print()

