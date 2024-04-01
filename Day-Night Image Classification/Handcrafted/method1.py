# Inspired by https://github.com/jayeshsaita/Day-Night-Classifier base model classifier
# Method description: Day-Night Classifier based on brightness (value) thresholding on HSV color space

import torch
from torcheval.metrics.functional import multiclass_confusion_matrix
from tqdm import tqdm

def get_avg_brightness(device, dataloader, dataset_size):
    
    brightness_sum = 0.0

    with tqdm(dataloader, unit="batch") as t:

        t.set_description(f"get treshold")
        
        for inputs, _ in t:
            
            inputs = inputs.to(device)
            brightness_sum += inputs.mean(dim=[2,3], dtype=torch.float).sum(dim=0)[2]

    return brightness_sum/dataset_size

    
def classify_dataset(device, dataloaders, class_names, dataset_sizes, threshold):

    for phase in ['train', 'val']:
        if phase == "val":
            confusion_matrix = torch.zeros((len(class_names), len(class_names))).to(device)

        running_corrects = 0

        with tqdm(dataloaders[phase], unit="batch") as t:

            t.set_description(f"{phase} classification")
            
            for inputs, targets in t:
                inputs, targets = inputs.to(device), targets.to(device)
                preds = (inputs.mean(dim=[2,3], dtype=torch.float)[:,2] < threshold).int()
                
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
