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

os.environ["WDM_PROGRESS_BAR"] = "0"

def train_model(device, model, dataloaders, optimizer, loss_function, early_stop, epochs, n_classes, dataset_sizes):
    
    torch.cuda.empty_cache()    
    model.to(device)

    best_model_path = os.path.join(args.save, f'{args.model}_best.pt')
    torch.save(model.state_dict(), best_model_path)
    
    acc = MulticlassAccuracy(num_classes=n_classes, device=device)
    
    best_epoch = 0
    best_acc = 0.0
    
    stop = False
        
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
                    one_hot_targets = nn.functional.one_hot(targets, num_classes=n_classes).float()
                    inputs, targets, one_hot_targets = inputs.to(device), targets.to(device), one_hot_targets.to(device)
                
                    # Zero the parameter gradients (similar to optimizer.zero_grad(), see Pytorch docs for details)
                    for param in model.parameters():
                        param.grad = None

                    with torch.set_grad_enabled(phase == 'train'):
                        
                        outputs = model(inputs)
                        loss = loss_function(outputs, one_hot_targets)
                        
                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                        
                        acc.update(outputs, targets)
                            
                    running_loss += loss.item() * inputs.size(0)
                    
                epoch_loss = running_loss / dataset_sizes[phase]
                
                # deep copy the model
                if phase == 'val':
                        
                    print(f"lr: {optimizer.param_groups[0]['lr']}")
                    
                    if float(acc.compute()) > best_acc:
                        best_acc = float(acc.compute())
                        best_epoch = epoch
                        torch.save(model.state_dict(), best_model_path)
                                        
                    stop = early_stop.check(epoch_loss)
                    scheduler.step(epoch_loss)

                print_metrics_v1(phase, epoch, acc, epoch_loss)

        if stop:
            print(f"early stopped at epoch: {epoch}")
            #torch.save({
            #    'epoch': epoch,
            #    'model_state_dict': model.state_dict(),
            #    'optimizer_state_dict': optimizer.state_dict(),
            #    'loss': epoch_loss,
            #    }, os.path.join(args.save, f'{args.model}_checkpoint_e{epoch}.pt'))
            #break
                    
    print(f'best val acc: {best_acc:4f} - achieved on epoch: {best_epoch]')
    #model.load_state_dict(torch.load(best_model_params_path))
    #return model            

if __name__ == "__main__":
    
    torch.set_num_threads(1)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--device")
    parser.add_argument("--root")
    parser.add_argument("--model")
    parser.add_argument("--save")
    parser.add_argument("--seed")
    parser.add_argument("--epoch", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--pretrained", default=True, action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    
    if args.seed is None:
        args.seed = datetime.now().timestamp()
        
    print(f"seed - {args.seed}")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    data_transforms = {
        'train': Transform(Compose([
            Resize(224,224),
            Normalize(mean=[0.27850261, 0.28775489, 0.29626652], std=[0.26558841, 0.26848849, 0.2683816]),
            #Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            Rotate(limit=15, p=0.4),
            VerticalFlip(p=0.5),
            RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.4),
            Blur(p=0.3),
            ToTensorV2()
        ])),
        'val': Transform(Compose([
            Resize(224,224),
            #Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            Normalize(mean=[0.27850261, 0.28775489, 0.29626652], std=[0.26558841, 0.26848849, 0.2683816]),
            ToTensorV2()
        ])),
    }    
    
    image_datasets = {
            x: torchvision.datasets.ImageFolder(os.path.join(args.root, x), 
                                                data_transforms[x], 
                                                loader=Transform.open_img) 
            
            for x in ['train', 'val']
    }
    
    dataloaders = {
            x: torch.utils.data.DataLoader( image_datasets[x], 
                                            batch_size=args.batch_size, 
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
    n_classes = len(class_names)

    model = get_model(args.model, n_classes, args.pretrained)

    #optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.00001)
    #loss_function = nn.CrossEntropyLoss()
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20)
    #early_stop = EarlyStop(patience=45)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.00001)
    loss_function = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, eps=1e-09)
    early_stop = EarlyStop(patience=25)
    
    train_model(device, model, dataloaders, optimizer, loss_function, early_stop, args.epoch, n_classes, dataset_sizes)
