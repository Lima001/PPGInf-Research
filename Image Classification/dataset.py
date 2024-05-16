import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

class CustomDatasetCSV(Dataset):
   
    def __init__(self, annotations, root, transform=None, target_transform=None):
        self.annotations = pd.read_csv(annotations)
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image_path = os.path.join(self.root, self.annotations.iloc[idx, 0])
        image = np.array(Image.open(image_path))
        
        label = self.annotations.iloc[idx, 1]
        
        if self.transform:
            image = self.transform(image=image)['image']
        
        if self.target_transform:
            label = self.target_transform(label)
        
        return image, label

class CustomDatasetDummyCSV(Dataset):
    
    def __init__(self, file, root):
        self.file = pd.read_csv(file)
        self.root = root
        
    def __len__(self):
        return len(self.file)

    def __getitem__(self, idx):   
        return self.file.iloc[idx,0], "-1"
