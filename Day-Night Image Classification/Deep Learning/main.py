import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
from torcheval.metrics import *
from torch.utils.data import DataLoader
from albumentations import *
from albumentations.pytorch import ToTensorV2
from tempfile import TemporaryDirectory
from PIL import Image
from tqdm import tqdm

from earlystop import *

os.environ["WDM_PROGRESS_BAR"] = "0"

EPOCHS = 50
BATCH_SIZE = 4
LR = 1e-4
#STEP_SIZE = 5
#GAMMA = 0.1

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

def print_metrics(phase, epoch, acc, loss, confusion_matrix, precision, recall, f1_score, auroc, num_classes):
    
    print(f"{phase} - epoch {epoch} - acc: {acc.compute()} - loss: {loss}")
    
    if phase == "val":
        
        print(f"{phase} - epoch {epoch} - precision: {precision.compute()}")
        print(f"{phase} - epoch {epoch} - recall: {recall.compute()}")
        print(f"{phase} - epoch {epoch} - f1_score: {f1_score.compute()}")
        print(f"{phase} - epoch {epoch} - auroc: {auroc.compute()}")
        print(f"{phase} - epoch {epoch} - confusion_matrix:\n{confusion_matrix.compute().int()}")
    
    print()

    acc.reset()
    confusion_matrix.reset()
    precision.reset()
    recall.reset()
    f1_score.reset()
    auroc.reset()
    

def train_model(device, model, dataloaders, optimizer, loss_function, early_stop, epochs, class_names, dataset_sizes):
    
    stop = False
    torch.cuda.empty_cache()
    
    # Transfer model to GPU
    model.to(device)
    
    task = "multiclass"
    if len(class_names) == 2:
        task = "binary"
    
    acc = MulticlassAccuracy(num_classes=len(class_names), device=device)
    confusion_matrix = MulticlassConfusionMatrix(num_classes=len(class_names), normalize=None, device=device)
    precision = MulticlassPrecision(num_classes=len(class_names), device=device, average=None)
    recall = MulticlassRecall(num_classes=len(class_names), device=device, average=None)
    f1_score = MulticlassF1Score(num_classes=len(class_names), device=device, average=None)
    auroc = MulticlassAUROC(num_classes=len(class_names), device=device, average=None)
    
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

            running_loss = 0.0
            
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
                        #_, preds = torch.max(outputs, 1)
                        loss = loss_function(outputs, one_hot_targets)
                        
                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                        
                        acc.update(outputs, targets)
                        confusion_matrix.update(outputs, targets)
                        precision.update(outputs, targets)
                        recall.update(outputs, targets)
                        f1_score.update(outputs, targets)
                        auroc.update(outputs, targets)
                            
                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                
                epoch_loss = running_loss / dataset_sizes[phase]
                
                # deep copy the model
                if phase == 'val':
                        
                    print(f"lr: {optimizer.param_groups[0]['lr']}")
                    
                    if float(acc.compute()) > best_acc:
                        best_acc = float(acc.compute())
                        torch.save(model.state_dict(), best_model_params_path)
                        
                    stop = early_stop.check(float(acc.compute()))
                    scheduler.step(epoch_loss)

                print_metrics(phase, epoch, acc, epoch_loss, confusion_matrix, precision, recall, f1_score, auroc, len(class_names))
                
        if stop:
            print(f"Early stopped at epoch {epoch}")
            break
                    
    print(f'Best val acc: {best_acc:4f}')    
    model.load_state_dict(torch.load(best_model_params_path))
    
    return model            

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

    resize_dim = 224
    if args.model == "inception_v3":
        resize_dim = 299    

    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': Transform(Compose([
            Resize(resize_dim,resize_dim),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            Rotate(limit=15, p=0.5),
            HorizontalFlip(p=0.5),
            RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.2),
            Blur(p=0.2),
            ToTensorV2()
        ])),
        'val': Transform(Compose([
            Resize(resize_dim,resize_dim),
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
    #print(dataset_sizes)

    class_names = image_datasets['train'].classes
    #print(class_names)

    if args.model == "vgg16":
        model = get_vgg16()
    elif args.model == "resnet18":
        model = get_resnet18()
    elif args.model == "inception_v3":
        model = get_inception_v3()
    elif args.model == "customCNN":
        model = get_customCNN()

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_function = nn.CrossEntropyLoss()
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    early_stop = EarlyStop()

    train_model(device, model, dataloaders, optimizer, loss_function, early_stop, EPOCHS, class_names, dataset_sizes)
