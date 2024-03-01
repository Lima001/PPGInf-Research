import os
import pandas as pd
from torch import float32
from torch.utils.data import Dataset

import cv2

class CustomDataset(Dataset):
    def __init__(self, device, annotations_file, img_dir, transform=None, target_transform=None):
        self.device = device
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = cv2.imread(image_path)
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)        
        
        label = self.img_labels.iloc[idx, 1]
        
        if self.transform:
            image = self.transform(image=image)['image']
        
        if self.target_transform:
            label = self.target_transform(label)
        
        return image, label