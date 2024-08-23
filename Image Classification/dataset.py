# This module defines custom dataset classes for loading and processing images and their corresponding labels 
# from CSV files and directories. It includes a dataset class for loading images specified in a CSV file 
# (with support for transformations), a dummy dataset class for handling CSVs with placeholder labels, 
# and a custom ImageFolder class that returns the image file paths along with the image and label.

import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torch
from torchvision import datasets

# Custom dataset class for loading images and labels from a CSV file
class CustomDatasetCSV(Dataset):
   
    def __init__(self, annotations, root, transform=None, target_transform=None):
        """
            Initializes the dataset with the path to the CSV file, the root directory for images,
            and optional transformations for images and labels.

            Args:
                annotations (str): Path to the CSV file containing image filenames and labels.
                root (str): Root directory where images are stored.
                transform (callable, optional): Optional transform to be applied to images. See transform.py
                target_transform (callable, optional): Optional transform to be applied to labels.
        """
        
        self.annotations = pd.read_csv(annotations)     # Load CSV file containing image filename and labels
        self.root = root                                # Root directory containing images
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        """
            Returns the total number of samples in the dataset.
        """
        return len(self.annotations)

    def __getitem__(self, idx):
        """
            Retrieves the image and label at the specified index.

            Args:
                idx (int): Index of the sample to retrieve.

            Returns:
                tuple: (image, label) where image is the (transformed) image and label is the corresponding class annotation.
        """
        
        # Get the image path, open it, and convert it to numpy array
        image_path = os.path.join(self.root, self.annotations.iloc[idx, 0])
        image = np.array(Image.open(image_path))
        
        # Get the label for the image
        label = self.annotations.iloc[idx, 1]
        
        # Apply transformations if provided
        if self.transform:
            image = self.transform(image=image)['image']
        if self.target_transform:
            label = self.target_transform(label)
        
        return image, label

# Custom dataset class for handling CSV files with dummy labels
class CustomDatasetDummyCSV(Dataset):
    
    def __init__(self, file, root):
        """
            Initializes the dataset with the path to the CSV file and the root directory for images.

            Args:
                file (str): Path to the CSV file containing image paths.
                root (str): Root directory where images are stored.
        """
        self.file = pd.read_csv(file)
        self.root = root
        
    def __len__(self):
        """
            Returns the total number of samples in the dataset.
        """
        return len(self.file)

    def __getitem__(self, idx):
        """
            Retrieves the image path and a dummy label for the specified index.

            Args:
                idx (int): Index of the sample to retrieve.

            Returns:
                tuple: (image_path, "-1") where "-1" is a dummy placeholder for the label.
        """   
        return self.file.iloc[idx,0], "-1"

# Custom ImageFolder class that returns image file paths along with images and labels.
# Reference: https://gist.github.com/andrewjong/6b02ff237533b3b2c554701fb53d5c4d
class ImageFolderWithPaths(datasets.ImageFolder):
    """
        Custom dataset that includes image file paths. Extends torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        """
            Overrides the __getitem__ method to return image file paths along with images and labels.

            Args:
                index (int): Index of the sample to retrieve.

            Returns:
                tuple: (image, label, path) where path is the file path of the image.
        """
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        
        # the image file path
        path = self.imgs[index][0]
        
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        
        return tuple_with_path
