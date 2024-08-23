# This script evaluates a deep learning model by printing the predicted probabilities for each image in a specified dataset subset. 
# The probabilities are computed using a softmax function applied to the model's output logits. 

import argparse
import torch
import torch.nn as nn
from albumentations import *
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from models import *
from dataset import *
from transform import *

def print_probs(device, model, dataloader):
    
    # Clear GPU cache to ensure there is no leftover data
    torch.cuda.empty_cache()
    
    # Move model to specified device (GPU or CPU)
    model.to(device)
    
    # Set model to evaluate mode
    model.eval()

    # Define a softmax function to transform the model output logit vector to a probability vector
    f = nn.Softmax()

    # tqdm is used for displaying progress bar
    with tqdm(dataloader, unit="batch", disable=True) as tepoch:

        # Iterate over batches of data from dataset
        for inputs, targets, paths in tepoch:

            # Move data to device (necessary to move from CPU to GPU)
            inputs = inputs.to(device)

            # Do not compute gradients, as the backward pass is not necessary
            with torch.set_grad_enabled(False):
                outputs = model(inputs)     # Forward pass
                
                # Iterate over each output prediction
                for i in range(outputs.size(dim=0)):
                    # Print the filename and the respective output probability vector 
                    print(f"{paths[i]}: {f(outputs[i]).cpu().tolist()}")
                

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
    parser.add_argument("--pretrained", default=False, action=argparse.BooleanOptionalAction) # Whether to use a pretrained model from PyTorch (ImageNet)
    parser.add_argument("--subset")                     # Subset (train/val/test) to load and calculate the metrics
    args = parser.parse_args()  
    
   # Determine device to use (GPU or CPU)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

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
    
    # Load dataset specified subset
    # Note: The ImageFolderWithPaths class is an extension of torchvision.datasets.ImageFolder
    #       See datasets.py for more details.
    image_dataset = ImageFolderWithPaths(args.root, 
                                         data_transforms[args.subset], 
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
    model = get_model(args.model, n_class, args.pretrained)
    model.load_state_dict(torch.load(args.config, map_location=device))
    
    print_probs(device, model, dataloader)