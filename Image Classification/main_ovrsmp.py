# Similar to main.py. However, this program is intended for fine-grained classification tasks 
# with an emphasis on handling imbalanced datasets. Oversammpling is done by increasing the
# dataloader frequency load for the minority classes.

# In a near future, main.py and main_ovrsmp.py will be merged together

import os
import random
import argparse
import torch
import torchvision
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from albumentations import *
from torcheval.metrics import *
from tqdm import tqdm
from datetime import datetime
from models import *
from dataset import *
from earlystop import *
from transform import *
from metrics import *


def train_model(device, model, dataloaders, optimizer, loss_function, early_stop, epochs, n_classes, dataset_sizes):
    
    # Clear GPU cache to ensure there is no leftover data
    torch.cuda.empty_cache()    
    
    # Move model to specified device (GPU or CPU)
    model.to(device)                

    # Path to save the best model during training
    best_model_path = os.path.join(args.save, f'{args.model}_best.pt')
    
    # Initial save of model state.
    torch.save(model.state_dict(), best_model_path)                         
    
    # Initialize accuracy metric for multiclass classification
    acc = MulticlassAccuracy(num_classes=n_classes, device=device)
    
    # Track the best epoch and accuracy
    best_epoch = 0                                  
    best_acc = 0.0
    
    # Flag for early stopping
    stop = False
        
    # Loop over the specified number of epochs
    for epoch in range(1, epochs+1):

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            # Initialize running loss for the epoch
            running_loss = 0.0
            
            # tqdm is used for displaying progress bar
            with tqdm(dataloaders[phase], unit="batch") as tepoch:
            
                # Iterate over batches of data
                for inputs, targets in tepoch:
                    tepoch.set_description(f"{phase} epoch {epoch}")    # Set description for progress bar
                
                    # Convert targets to one-hot encoding
                    one_hot_targets = nn.functional.one_hot(targets, num_classes=n_classes).float()
                    
                    # Move data to device (necessary to move from CPU to GPU)
                    inputs, targets, one_hot_targets = inputs.to(device), targets.to(device), one_hot_targets.to(device)
                
                    # Zero the parameter gradients; similar to optimizer.zero_grad(), see Pytorch docs for details
                    for param in model.parameters():
                        param.grad = None

                    # Forward pass and backward pass (only in training phase)
                    with torch.set_grad_enabled(phase == 'train'):
                        
                        outputs = model(inputs)     # Forward pass
                        loss = loss_function(outputs, one_hot_targets) # Calculate loss given a loss function
                        
                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()     # Backpropagation
                            optimizer.step()    # Update the optimizer parameters (if conditions are met)
                        
                        # Update accuracy metric with batch results
                        acc.update(outputs, targets)
                            
                    # Accumulate loss using batch data
                    running_loss += loss.item() * inputs.size(0)
                    
                # Calculate average loss for the epoch
                epoch_loss = running_loss / dataset_sizes[phase]
                
                if phase == 'val':
                    
                    # Print current learning rate - usefull when using lr scheduler
                    print(f"lr: {optimizer.param_groups[0]['lr']}")
                    
                    # Check if the current model is the best so far 
                    if float(acc.compute()) > best_acc:
                        
                        # Update best model summary info
                        best_acc = float(acc.compute())
                        best_epoch = epoch
                        
                        # Save the best model
                        torch.save(model.state_dict(), best_model_path)
                     
                    # Check if early stopping should be triggered                   
                    stop = early_stop.check(epoch_loss)
                    
                    # Update the learning rate scheduler
                    scheduler.step(epoch_loss)

                # Print metrics for the current epoch
                print_metrics_v1(phase, epoch, acc, epoch_loss)

        # Stop training if early stopping criteria are met
        if stop:
            print(f"early stopped at epoch: {epoch}")
            
            # Uncomment to save the last model (note that it is not necessarily the best performing model)
            #torch.save({
            #    'epoch': epoch,
            #    'model_state_dict': model.state_dict(),
            #    'optimizer_state_dict': optimizer.state_dict(),
            #    'loss': epoch_loss,
            #    }, os.path.join(args.save, f'{args.model}_checkpoint_e{epoch}.pt'))
            
            break
         
    # Print the best validation accuracy and the epoch it was achieved on           
    print(f'best val acc: {best_acc:4f} - achieved on epoch: {best_epoch}')
    
    # Uncomment to load and return the best performing model
    #model.load_state_dict(torch.load(best_model_params_path))
    #return model            

if __name__ == "__main__":
    
    # Uncomment to set PyTorch as single thread
    #torch.set_num_threads(1)
    
    # Argument parser for command-line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("--device")                 # Specify device (e.g., 'cuda:0' or 'cpu').
    parser.add_argument("--root")                   # Root directory for the dataset (where train/test/val folders are)
    parser.add_argument("--model")                  # Model architecture to use (see models.py for a list of supported models)
    parser.add_argument("--save")                   # Directory to save the best model
    parser.add_argument("--seed")                   # Seed for random number generators
    parser.add_argument("--epoch", type=int)        # Max number of epochs
    parser.add_argument("--batch_size", type=int)   # Batch size for data loaders
    parser.add_argument("--lr", type=float)         # Initial learning rate for the optimizer
    parser.add_argument("--pretrained", default=False, action=argparse.BooleanOptionalAction)   # Whether to use a pretrained model from PyTorch (ImageNet)
    args = parser.parse_args()
    
    # Set random seed if not provided
    if args.seed is None:
        args.seed = datetime.now().timestamp()
    
    # Log: print seed for random number generators
    print(f"seed - {args.seed}")
    torch.manual_seed(args.seed)    # Set seed for PyTorch lib
    random.seed(args.seed)          # Set seed for random lib
    
    # Determine device to use (GPU or CPU)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    # Log: print whether pretrained model is used
    print(args.pretrained)

    # If pretrained models are being used, define mean and standard deviation for normalization (based on ImageNet dataset)
    if args.pretrained:
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    
    # Else, you should inform the values to be used (normalization.py can be used and will be further integrated with this code)
    # For now the deafult behaviour is to quit the code
    else:
        exit(0)

    # Define data augmentation and normalization transforms
    data_transforms = {
        'train': Transform(Compose([
            Resize(224,224),
            Affine(scale=(0.9,1.3), rotate=(-360,360), shear=(-30,30), p=0.5),
            RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
            AdvancedBlur(p=0.4),
            CoarseDropout(max_holes=1, max_height=72, max_width=72, min_holes=1, min_height=72, min_width=72, fill_value="random", p=0.25),
            Normalize(mean=mean, std=std),
            ToTensorV2()
        ])),
        'val': Transform(Compose([
            Resize(224,224),
            Normalize(mean=mean, std=std),
            ToTensorV2()
        ])),
    } 
    
    # Load the datasets with the defined transformations
    image_datasets = {
            x: torchvision.datasets.ImageFolder(os.path.join(args.root, x), 
                                                data_transforms[x],
                                                loader=Transform.open_img) 
            
            for x in ['train', 'val']
    }
   

    # Get class names and number of classes
    class_names = image_datasets['train'].classes
    n_classes = len(class_names)

    # Calculate class weights to handle class imbalance
    samples_per_class = torch.zeros(n_classes, dtype=torch.long)
    
    for _, target in image_datasets['train']:
        samples_per_class[target] += 1

    # The higher the class presente in the dataset, the lower is its loading frequence during training
    class_weights = 1/samples_per_class
    sample_weights = [class_weights[i] for i in image_datasets["train"].targets]

    # Create data loaders with weighted sampling for the training set
    dataloaders = {
            "train": torch.utils.data.DataLoader(image_datasets["train"], 
                                                 batch_size=args.batch_size, 
                                                 shuffle=False, 
                                                 num_workers=2,
                                                 pin_memory=True,
                                                 sampler=WeightedRandomSampler(weights=sample_weights, num_samples=len(image_datasets["train"]), replacement=True)), 
            
            "val": torch.utils.data.DataLoader(image_datasets["val"], 
                                                 batch_size=args.batch_size, 
                                                 shuffle=True, 
                                                 num_workers=2,
                                                 pin_memory=True)
    }
    
    # Get dataset sizes
    dataset_sizes = {
            x: len(image_datasets[x]) 
            for x in ['train', 'val']
    }
    
    model = get_model(args.model, n_classes, args.pretrained)

    # Set up optimizer, loss function, learning rate scheduler, and early stopping mechanism
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.00001)
    loss_function = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, eps=1e-09)
    early_stop = EarlyStop(patience=15)
    
    # Start training the model
    train_model(device, model, dataloaders, optimizer, loss_function, early_stop, args.epoch, n_classes, dataset_sizes)
