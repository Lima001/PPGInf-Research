import os
import sys
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torcheval.metrics.functional import multiclass_confusion_matrix
from albumentations import *
from albumentations.pytorch import ToTensorV2
from tempfile import TemporaryDirectory
from PIL import Image
from tqdm import tqdm

os.environ["WDM_PROGRESS_BAR"] = "0"

EPOCHS = 5
BATCH_SIZE = 4
LR = 1e-4
STEP_SIZE = 2
GAMMA = 0.1

class Transform():
    
    def __init__(self,transform):
        self.transform=transform
    
    def __call__(self,image):
        return self.transform(image=image)["image"]
    
    @staticmethod
    def open_img(img_path):
        img = Image.open(img_path)
        return np.array(img)

def get_vgg16():

    model = torchvision.models.vgg16(weights='DEFAULT')

    for param in model.parameters():
        param.requires_grad = False

    num_features = model.classifier[6].in_features
    features = list(model.classifier.children())[:-1]
    features.extend([nn.Linear(num_features, len(class_names))])
    model.classifier = nn.Sequential(*features)
    nn.init.xavier_uniform_(model.classifier[6].weight)
    nn.init.constant_(model.classifier[6].bias, 0.0)

    return model

def get_resnet18():
    
    model = torchvision.models.resnet18(weights='DEFAULT')

    for param in model.parameters():
        param.requires_grad = False

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, len(class_names))
    nn.init.xavier_uniform_(model.fc.weight)
    nn.init.constant_(model.fc.bias, 0.0)

    return model

def get_inception_v3():

    model = torchvision.models.inception_v3(weights='DEFAULT')

    for param in model.parameters():
        param.requires_grad = False

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, len(class_names))
    model.aux_logits = False
    model.AuxLogits = None
    nn.init.xavier_uniform_(model.fc.weight)
    nn.init.constant_(model.fc.bias, 0.0)

    return model

def train_model(device, model, dataloaders, optimizer, loss_function, epochs, class_names, dataset_sizes):

    torch.cuda.empty_cache()
    
    # Transfer model to GPU
    model.to(device)
    
    best_model_params_path = os.path.join(args.save, f'{args.model}_best_params.pt')
    torch.save(model.state_dict(), best_model_params_path)
    best_acc = 0.0
        
    # Train the model. In PyTorch we have to implement the training loop ourselves.
    for epoch in range(1, epochs+1):

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
                confusion_matrix = torch.zeros((len(class_names), len(class_names))).to(device)

            running_loss = 0.0
            running_corrects = 0

            with tqdm(dataloaders[phase], unit="batch") as tepoch:
            
                for inputs, targets in tepoch:
                    tepoch.set_description(f"{phase} epoch {epoch}")
                
                    # Move data to GPU.
                    one_hot_targets = nn.functional.one_hot(targets, num_classes=len(class_names)).float()
                    inputs, targets, one_hot_targets = inputs.to(device), targets.to(device), one_hot_targets.to(device)
                
                    # Zero the parameter gradients.
                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = loss_function(outputs, one_hot_targets)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                        else:
                            confusion_matrix += multiclass_confusion_matrix(targets, preds, len(class_names)).to(device)
                            
                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == targets)
                
                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print(f'{phase} epoch {epoch} - Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # deep copy the model
                if phase == 'val':
                        
                    if epoch_acc > best_acc:
                        best_acc = epoch_acc
                        torch.save(model.state_dict(), best_model_params_path)                    


                    print("Confusion matrix (raw counts)")
                    for i in range(len(class_names)):
                        for j in range(len(class_names)):
                            print(int(confusion_matrix[i][j]), end="\t")
                        print()
        
    print(f'Best val Acc: {best_acc:4f}')    
    model.load_state_dict(torch.load(best_model_params_path))
    
    return model            

def evaluate_model(device, model, dataloader, class_names, dataset_size):

    torch.cuda.empty_cache()
    
    # Transfer model to GPU
    model.to(device)
    model.eval()   
    
    confusion_matrix = torch.zeros((len(class_names), len(class_names))).to(device)
    running_loss = 0.0
    running_corrects = 0

    with tqdm(dataloader, unit="batch") as t:
    
        for inputs, targets in t:
            tepoch.set_description(f"evaluating")
        
            # Move data to GPU.
            one_hot_targets = nn.functional.one_hot(targets, num_classes=len(class_names)).float()
            inputs, targets, one_hot_targets = inputs.to(device), targets.to(device), one_hot_targets.to(device)
        
            # Zero the parameter gradients.
            optimizer.zero_grad()

            with torch.set_grad_enabled(False):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = loss_function(outputs, one_hot_targets)
                confusion_matrix += multiclass_confusion_matrix(targets, preds, len(class_names)).to(device)
                    
            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == targets)

        eval_loss = running_loss / dataset_size
        eval_corrects = running_corrects.double() / dataset_size

        print(f'Loss: {eval_loss:.4f} Acc: {eval_corrects:.4f}')
                 
        print("Confusion matrix (raw counts)")
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                print(int(confusion_matrix[i][j]), end="\t")
            print()


def visualize_model(device, model, dataloader, num_images=6):

    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(f'predicted: {class_names[preds[j]]}')
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    return

def model_prediction(device, model, data_transform, img_path, graphics=False):
    
    model.eval()

    img = Image.open(img_path)
    img = data_transforms(img)
    img = img.unsqueeze(0)
    img = img.to(device)

    with torch.no_grad():
        outputs = model(img)
        _, preds = torch.max(outputs, 1)

        if graphics:
            ax = plt.subplot(2,2,1)
            ax.axis('off')
            ax.set_title(f'Predicted: {class_names[preds[0]]}')
            imshow(img.cpu().data[0])

    return preds[0]

if __name__ == "__main__":
    
    torch.set_num_threads(1)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--device")
    parser.add_argument("--data_dir")
    parser.add_argument("--model")
    parser.add_argument("--save")
    parser.add_argument("--seed")
    args = parser.parse_args()
    
    if args.seed is None:
        args.seed = 0

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)    
    
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': Transform(Compose([
            Resize(224,224),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            Rotate(limit=15, p=0.5),
            HorizontalFlip(p=0.5),
            RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.2),
            Blur(p=0.2),
            ToTensorV2()
        ])),
        'val': Transform(Compose([
            Resize(224,224),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])),
    }    
    
    image_datasets = {
            x: torchvision.datasets.ImageFolder(os.path.join(args.data_dir, x), 
                                                data_transforms[x], 
                                                loader=Transform.open_img) 
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
    #print(class_names)

    if args.model == "vgg16":
        model = get_vgg16()
    elif args.model == "resnet18":
        model = get_resnet18()
    elif args.model == "inception_v3":
        model = get_inception_v3()

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_function = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

    train_model(device, model, dataloaders, optimizer, loss_function, EPOCHS, class_names, dataset_sizes)
