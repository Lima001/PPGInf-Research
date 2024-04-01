# Based on :
#   Saha, B.; Davies, D.; Raghavan, A. 
#   Day Night Classification of Images Using Thresholding on HSV Histogram. 
#   U.S. Patent 9,530,056, 27 Dezembro 2013; 
# classification method
# Method description: Day-Night Classifier based on HSV histogram thresholding

import torch
from torcheval.metrics.functional import multiclass_confusion_matrix
from tqdm import tqdm

HUE_UPPER = 288
HUE_LOWER = 72
VALUE = 150

def get_avg_thresholds(device, dataloader, dataset_size):
    
    nh_sum = 0.0
    nv_sum = 0.0

    with tqdm(dataloader, unit="batch") as t:

        for inputs, _ in t:
            inputs = inputs.to(device)    
            inputs = inputs.float()

            #inputs[:,0,:,:] /= 179.0
            #inputs[:,2,:,:] /= 255
                
            nh_partial1 = (inputs[:,0,:,:] >= HUE_UPPER).int().sum(dim=[1,2,0])
            nh_partial2 = (inputs[:,0,:,:] >= HUE_LOWER).int().sum(dim=[1,2,0])
            nh_sum += nh_partial1 + nh_partial2
            nv_sum += (inputs[:,2,:,:] >= VALUE).int().sum(dim=[1,2,0])

    return (nh_sum/dataset_size, nv_sum/dataset_size)

def classify_dataset(device, dataloaders, class_names, dataset_sizes, threshold_h, threshold_v):

    for phase in ['train', 'val']:
        if phase == "val":
            confusion_matrix = torch.zeros((len(class_names), len(class_names))).to(device)

        running_corrects = 0

        with tqdm(dataloaders[phase], unit="batch") as t:

            for inputs, targets in t:
                inputs, targets = inputs.to(device), targets.to(device)
                inputs = inputs.float()

                nh_partial1 = (inputs[:,0,:,:] >= HUE_UPPER).int().sum(dim=[1,2])
                nh_partial2 = (inputs[:,0,:,:] >= HUE_LOWER).int().sum(dim=[1,2])
                nh_tensor = nh_partial1 + nh_partial2
                nv_tensor = (inputs[:,2,:,:] >= VALUE).int().sum(dim=[1,2])

                preds = ((nh_tensor > threshold_h).bool() & (nv_tensor < threshold_v).bool()).int()

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
