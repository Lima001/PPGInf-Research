import os
import argparse
import torch
import torch.nn as nn
from albumentations import *
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from models import *
from dataset import *
from transform import *

# Prediction directory name
OUTPUT = "Predicts"

def save_predicts(device, model, dataloader):

    # Define the absolute path to predictions directory root
    abs_path = os.path.join(os.path.abspath("./"), OUTPUT)

    # Clear GPU cache to ensure there is no leftover data
    torch.cuda.empty_cache()
    
    # Move model to specified device (GPU or CPU)
    model.to(device)
    
    # Set model to evaluate mode
    model.eval()

    # tqdm is used for displaying progress bar
    with tqdm(dataloader, unit="batch") as tepoch:

        # Iterate over batches of data from dataset
        for inputs, targets, paths in tepoch:

            # Move data to device (necessary to move from CPU to GPU)
            inputs = inputs.to(device)

            # Do not compute gradients, as the backward pass is not necessary
            with torch.set_grad_enabled(False):
                outputs = model(inputs)             # Forward pass
                _, preds = torch.max(outputs, 1)    # Get top-1 prediction class
                
                # Iterate over each output prediction
                for i in range(outputs.size(dim=0)):
                    # Define where to save the image
                    aux = os.path.join(abs_path, f"{int(preds[i].cpu())}", f"{paths[i].split('/')[-1]}")
                    
                    # As I am using Linux, to save space the confusion matrix directory is composed of symbolic links
                    os.system(f"ln -s {paths[i]} {aux}")

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
    parser.add_argument("--n_class", type=int)       
    parser.add_argument("--pretrained", default=False, action=argparse.BooleanOptionalAction) # Whether to normalized based on ImageNet mean and std
    args = parser.parse_args()

    abs_path = os.path.join(os.path.abspath("./"), OUTPUT)
    if not os.path.exists(abs_path):
        os.makedirs(abs_path)
    else:
        print(f"{OUTPUT} directory already exists. Please remove it (or move it to another path)")
        exit(0)
    
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
    data_transforms = Transform(Compose([
                            Resize(224,224),
                            Normalize(mean=mean, std=std),
                            ToTensorV2()
    ]))
    
    # Load image set
    image_dataset = ImageFolderWithPaths(args.root, 
                                         data_transforms, 
                                         loader=Transform.open_img)

    # Create data loaders for the subset
    dataloader = torch.utils.data.DataLoader(image_dataset, 
                                             batch_size=args.batch_size, 
                                             shuffle=False, 
                                             num_workers=0,
                                             pin_memory=False)
    
    # Get subset summary info
    dataset_size = len(image_dataset) 

    # Create the subdirectories on OUTPUT directory
    for i in range(args.n_class):
        abs_path = os.path.join(os.path.abspath("./"), OUTPUT, str(i))
        
        if not os.path.exists(abs_path):
            os.makedirs(abs_path)
        else:
            print(f"Directory already exists!")
            exit(0)

    # Initialize the model (see model.py)
    model = get_model(args.model, args.n_class, args.pretrained)
    model.load_state_dict(torch.load(args.config, map_location=device))
        
    save_predicts(device, model, dataloader)
