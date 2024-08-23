# This script evaluates a deep learning model's performance on a dataset. It calculates and prints
# various metrics including accuracy, precision, recall, F1 score, and confusion matrix.
# Note that metrics are from torcheval.metrics

import os
import argparse
import torch
import torch.nn as nn
import torchvision
from torcheval.metrics import *
from albumentations import *
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from models import *
from dataset import *
from earlystop import *
from transform import *
from metrics import *

# Prints evaluation metrics for a single phase (train/validation) and epoch
def print_metrics_v1(phase, epoch, acc, loss):

    print(f"{phase} - epoch {epoch} - acc: {acc.compute()} - loss: {loss}")
    # Reset accuracy metric, otherwise results from previous epochs will be also computed in the next call of this function
    acc.reset()
    
def print_metrics_v2(global_acc, local_acc, top_2_acc, confusion_matrix, precision, recall, f1_score):
    
    print(f"global acc: {global_acc.compute()}")        
    print(f"local acc: {local_acc.compute()}")        
    print(f"top 2 acc: {top_2_acc.compute()}")        
    print(f"precision: {precision.compute()}")
    print(f"recall: {recall.compute()}")
    print(f"f1_score: {f1_score.compute()}")
    print(f"confusion_matrix:\n{confusion_matrix.compute().int()}")
    print()

    # Reset metrics, otherwise results from previous epochs will be also computed in the next call of this function
    global_acc.reset()
    local_acc.reset()
    top_2_acc.reset()
    confusion_matrix.reset()
    precision.reset()
    recall.reset()
    f1_score.reset()

def get_metrics(device, model, dataloader, loss_function, n_class, dataset_size):
    """
        Evaluates the model on the dataset calculating a set of metrics and printing them on screen.
        
        Args:
            device (torch.device): The device to perform computation on (CPU or GPU).
            model (torch.nn.Module): The model to be evaluated.
            dataloader (torch.utils.data.DataLoader): DataLoader providing the dataset.
            loss_function (torch.nn.Module): Loss function used for evaluation.
            n_class (int): Number of classes in the dataset.
            dataset_size (int): Total number of samples in the dataset.
    """
    
    torch.cuda.empty_cache()
    model.to(device)    
    model.eval()

    # Note: 
    #   Macro averaging calculated by averaging all classes individual metric  
    #   K defines the top-k probabilities that should be considered when evaluating the model 

    # Initialize accuracy metrics 
    global_acc = MulticlassAccuracy(num_classes=n_class, device=device, average='macro')
    local_acc = MulticlassAccuracy(num_classes=n_class, device=device, average="macro")
    top_2_acc = MulticlassAccuracy(num_classes=n_class, device=device, average="macro", k=2)
    
    # Initialize auxiliar metrics
    confusion_matrix = MulticlassConfusionMatrix(num_classes=n_class, normalize=None, device=device)
    precision = MulticlassPrecision(num_classes=n_class, average="macro", device=device)
    recall = MulticlassRecall(num_classes=n_class, average="macro", device=device)
    f1_score = MulticlassF1Score(num_classes=n_class, average="macro", device=device)

    # Initialize running loss (incremented at each batch)
    running_loss = 0.0
    
    # tqdm is used for displaying progress bar
    with tqdm(dataloader, unit="batch") as tepoch:
    
        # Iterate over batches of data
        for inputs, targets in tepoch:
        
            # Convert targets to one-hot encoding
            one_hot_targets = nn.functional.one_hot(targets, num_classes=n_class).float()
            
            # Move data to device (necessary to move from CPU to GPU)
            inputs, targets, one_hot_targets = inputs.to(device), targets.to(device), one_hot_targets.to(device)

            # Do not compute gradients, as the backward pass is not necessary
            with torch.set_grad_enabled(False):
                outputs = model(inputs)                             # Forward pass
                loss = loss_function(outputs, one_hot_targets)      # Calculate loss given a loss function
                
                # Update metrics with batch results
                global_acc.update(outputs, targets)
                local_acc.update(outputs, targets)
                top_2_acc.update(outputs, targets)
                confusion_matrix.update(outputs, targets)
                precision.update(outputs, targets)
                recall.update(outputs, targets)
                f1_score.update(outputs, targets)
                
                # Accumulate loss using batch data
                running_loss += loss.item() * inputs.size(0)

        # Compute and print metrics
        # print(f"Loss: {running_loss / dataset_size}")
        print_metrics_v2(global_acc, local_acc, top_2_acc, confusion_matrix, precision, recall, f1_score)


if __name__ == "__main__":
    
    # Comment to set PyTorch as multi-thread
    torch.set_num_threads(1)
    
    # Argument parser for command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--device")                     # Specify device (e.g., 'cuda:0' or 'cpu')
    parser.add_argument("--root")                       # Root directory for the dataset (where train/test/val folders are)
    parser.add_argument("--model")                      # Model architecture to use (see models.py for a list of supported models)
    parser.add_argument("--config")                     # Model weights that will be loaded before the metric evaluation step
    parser.add_argument("--batch_size", type=int)       # Batch size for data loaders
    parser.add_argument("--subset")                     # Subset (train/val/test) to load and calculate the metrics
    args = parser.parse_args()  
    
    # Determine device to use (GPU or CPU)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    # Define data augmentation and normalization transform (based on ImageNet)
    data_transform = Transform(Compose([
            Resize(224,224),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
    ]))
    
    # Load dataset specified subset
    image_dataset = torchvision.datasets.ImageFolder(os.path.join(args.root, args.subset), 
                                                     data_transform, 
                                                     loader=Transform.open_img) 
    
    # Create data loaders for the subset
    dataloader = torch.utils.data.DataLoader(image_dataset, 
                                             batch_size=args.batch_size, 
                                             shuffle=False, 
                                             num_workers=0,
                                             pin_memory=False)
    
    # Get subset summary info
    dataset_size = len(image_dataset) 
    class_names = image_dataset.classes
    n_class = len(class_names)

    # Initialize the model (see model.py)
    model = get_model(args.model, n_class, False)
    model.load_state_dict(torch.load(args.config, map_location=device))
        
    # Set up the loss function for classification
    loss_function = nn.CrossEntropyLoss()
    
    # Start model evaluation
    get_metrics(device, model, dataloader, loss_function, n_class, dataset_size)
