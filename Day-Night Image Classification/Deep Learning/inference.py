import os
import sys
import cv2
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
#from torcheval.metrics.functional import multiclass_confusion_matrix
from albumentations import *
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from dataset import CustomDataset

os.environ["TQDM_DISABLE"] = "1"

BATCH_SIZE = 32
NUM_CLASSES = 2
PATH = 'config/vgg16.pth'

def perform_inference(device, model, transform, root_dir):
    torch.cuda.empty_cache()

    model.to(device)
    model.eval() 

    night_images = 0
    day_images = 0

    for filename in os.listdir(root_dir):
        
        if os.path.isfile(f"{root_dir}/{filename}"):
            
            image = image = cv2.imread(f"{root_dir}/{filename}")
            
            input_tensor = transform(image=image)['image']
            inputs = input_tensor.unsqueeze(0)
            inputs = inputs.to(device)
            
            with torch.no_grad():
                outputs = model(inputs)
                _, indices = torch.max(outputs, 1)
                day_images += indices[0] == 0
                night_images += indices[0] == 1
                
    return (day_images, night_images)

if __name__ == "__main__":

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    transform = Compose([
        Resize(224,224),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    model = torchvision.models.vgg16(weights='DEFAULT')
    num_features = model.classifier[6].in_features
    features = list(model.classifier.children())[:-1]
    features.extend([nn.Linear(num_features, NUM_CLASSES)])
    model.classifier = nn.Sequential(*features)
    model.load_state_dict(torch.load(PATH))
    
    result = perform_inference(device, model, transform, sys.argv[1])
    print(result)