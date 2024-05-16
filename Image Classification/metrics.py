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

os.environ["WDM_PROGRESS_BAR"] = "0"

def print_metrics_v1(phase, epoch, acc, loss):
    
    print(f"{phase} - epoch {epoch} - acc: {acc.compute()} - loss: {loss}")
    acc.reset()
    
def print_metrics_v2(acc, loss, confusion_matrix, precision, recall, ap, f1_score):
    
    print(f"acc: {acc.compute()} - loss: {loss}")        
    print(f"precision: {precision.compute()}")
    print(f"recall: {recall.compute()}")
    print(f"average precision: {ap.compute()}")
    print(f"f1_score: {f1_score.compute()}")
    print(f"confusion_matrix:\n{confusion_matrix.compute().int()}")
    print()

    acc.reset()
    confusion_matrix.reset()
    precision.reset()
    recall.reset()
    ap.reset()
    f1_score.reset()

def get_metrics(device, model, dataloader, loss_function, n_class, dataset_size):
    
    torch.cuda.empty_cache()
    model.to(device)    
    model.eval()
    
    acc = MulticlassAccuracy(num_classes=n_class, device=device)
    confusion_matrix = MulticlassConfusionMatrix(num_classes=n_class, normalize=None, device=device)
    precision = MulticlassPrecision(num_classes=n_class, average=None, device=device)
    recall = MulticlassRecall(num_classes=n_class, average=None, device=device)
    ap = MulticlassAUPRC(num_classes=n_class, average=None, device=device)
    f1_score = MulticlassF1Score(num_classes=n_class, average=None, device=device)

    running_loss = 0.0
    
    with tqdm(dataloader, unit="batch") as tepoch:
    
        for inputs, targets in tepoch:
        
            # Move data to GPU.
            one_hot_targets = nn.functional.one_hot(targets, num_classes=n_class).float()
            inputs, targets, one_hot_targets = inputs.to(device), targets.to(device), one_hot_targets.to(device)

            with torch.set_grad_enabled(False):
                outputs = model(inputs)
                loss = loss_function(outputs, one_hot_targets)
                
                acc.update(outputs, targets)
                confusion_matrix.update(outputs, targets)
                precision.update(outputs, targets)
                recall.update(outputs, targets)
                ap.update(outputs, targets)
                f1_score.update(outputs, targets)
                    
                running_loss += loss.item() * inputs.size(0)

        val_loss = running_loss / dataset_size
        print_metrics_v2(acc, val_loss, confusion_matrix, precision, recall, ap, f1_score)


if __name__ == "__main__":
    
    torch.set_num_threads(1)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--device")
    parser.add_argument("--root")
    parser.add_argument("--model")
    parser.add_argument("--config")
    parser.add_argument("--batch_size", type=int)
    args = parser.parse_args()  
    
    if args.device == "cpu":
        device = torch.device('cpu')
    else:
        device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    data_transform = Transform(Compose([
            Resize(224,224),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            #Normalize(mean=[0.27850261, 0.28775489, 0.29626652], std=[0.26558841, 0.26848849, 0.2683816]),
            ToTensorV2()
    ]))
    
    image_dataset = torchvision.datasets.ImageFolder(os.path.join(args.root, "train"), 
                                                     data_transform, 
                                                     loader=Transform.open_img) 
    
    dataloader = torch.utils.data.DataLoader(image_dataset, 
                                             batch_size=args.batch_size, 
                                             shuffle=False, 
                                             num_workers=0,
                                             pin_memory=False)
    
    dataset_size = len(image_dataset) 
    class_names = image_dataset.classes
    n_class = len(class_names)

    model = get_model(args.model, n_class, False)
    model.load_state_dict(torch.load(args.config, map_location=device))
        
    loss_function = nn.CrossEntropyLoss()
    get_metrics(device, model, dataloader, loss_function, n_class, dataset_size)
