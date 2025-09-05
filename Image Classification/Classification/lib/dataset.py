import csv
import os
import torch
from glob import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from ultralytics.data.augment import classify_augmentations, classify_transforms

class MultiTaskDataset(Dataset):
    """
    Custom PyTorch Dataset for multi-task image classification.

    Loads images and their corresponding labels for multiple tasks from a
    directory and a CSV file. The CSV is expected to have no header,
    with the first column being the image filename and subsequent columns
    being integer labels for each task.
    """
    
    def __init__(self, csv_path, img_dir, transform=None, task_names=None):
        """
        Initializes the dataset.

        Args:
            csv_path: Path to the CSV file (headerless);
            img_dir: Directory containing the image files;
            transform: A function/transform to apply to the images;
            task_names: An ordered list of task names. The order must match the label columns in the CSV;
        """
        
        if task_names is None:
            raise ValueError("`task_names` must be provided and match the CSV label columns.")
        
        self.img_dir = img_dir
        self.transform = transform
        self.task_names = task_names
        self.samples = []
        
        # Read the CSV file and store filename-label pairs.
        with open(csv_path, newline='') as f:
            reader = csv.reader(f)
            
            for row in reader:
                filename, labels_str = row[0], row[1:]
                
                # Skip rows with no labels
                if not labels_str:
                    continue
                
                labels = list(map(int, labels_str))
                
                if len(labels) != len(task_names):
                    raise ValueError("Mismatch between the number of labels and task_names.")
                
                self.samples.append((filename, labels))

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Retrieves an image and its labels at the specified index.

        Args:
            idx: The index of the item to retrieve.

        Returns:
            A tuple containing the transformed image and a dictionary of labels.
            If return_filename is True, the filename is also returned.
        """
        
        filename, labels = self.samples[idx]
        img_path = os.path.join(self.img_dir, filename)
        
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label_dict = {task: label for task, label in zip(self.task_names, labels)}

        return image, label_dict, filename
        

class InferenceDataset(Dataset):
    """
    Custom PyTorch Dataset for inference on a directory of images.

    Loads images from a specified directory without labels, which is useful for
    making predictions on a new pool of images.
    """
    def __init__(self, source_image_dir, transform=None):
        """
        Initializes the inference dataset.

        Args:
            source_image_dir: The directory containing images for inference.
            transform: An optional transform to apply to the images.
        """
        
        self.source_image_dir = source_image_dir
        self.transform = transform

        # Find all images with common extensions
        extensions = ["*.jpg", "*.png", "*.jpeg"]
        
        self.image_paths = []
        for ext in extensions:
            self.image_paths.extend(glob(os.path.join(source_image_dir, ext)))
        
        self.image_paths.sort()

    def __len__(self):
        """Returns the total number of images found."""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Loads and returns an image and its filename at the specified index.

        Args:
            idx: The index of the item to retrieve.

        Returns:
            A tuple containing the transformed image and its filename.
        """
        
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        filename = os.path.basename(img_path)

        if self.transform:
            image = self.transform(image)
        
        return image, None, filename

def multitask_collate_fn(batch):
    """
    Custom collate function for multi-task batches from 3-tuple datasets.
    
    This function handles batches where labels may or may not be present.
    """
    images, label_dicts, filenames = zip(*batch)
    images_tensor = torch.stack(images)
    
    # Check if the first label dictionary is not None.
    if label_dicts[0] is not None:
        task_keys = label_dicts[0].keys()
        batched_labels = {
            key: torch.tensor([d[key] for d in label_dicts], dtype=torch.long)
            for key in task_keys
        }
    else:
        # If labels are not present (inference mode), set batched_labels to None.
        batched_labels = None
    
    return images_tensor, batched_labels, filenames


def build_dataloaders(task_classes, csv_path, img_dir, batch_size, num_workers, fold_idx):
    """
    Builds and returns training and validation dataloaders using explicit parameters.
    """
    
    task_names = list(task_classes.keys())
    
    train_csv = os.path.join(csv_path, str(fold_idx), "train.txt")
    val_csv = os.path.join(csv_path, str(fold_idx), "val.txt")

    train_transform = classify_augmentations(auto_augment="randaugment")
    val_transform = classify_transforms()

    train_ds = MultiTaskDataset(train_csv, img_dir, transform=train_transform, task_names=task_names)
    val_ds = MultiTaskDataset(val_csv, img_dir, transform=val_transform, task_names=task_names)

    loader_params = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "collate_fn": multitask_collate_fn,
        "pin_memory": True,
        "persistent_workers": num_workers > 0,
    }

    train_loader = DataLoader(train_ds, shuffle=True, **loader_params)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_params)
    
    return train_loader, val_loader