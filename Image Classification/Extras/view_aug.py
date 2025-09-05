"""
A utility script to visualize the effects of image augmentations from the `ultralytics` library. 
It reads a specified number of images from a source directory, applies a series of random
augmentations to each, and saves the resulting images to an output directory for inspection.
"""
import argparse
import os
import random
from pathlib import Path
from typing import List

import torch
from PIL import Image
from torchvision.transforms import ToPILImage
from tqdm import tqdm

from ultralytics.data.augment import classify_augmentations

def get_image_paths(source_dir: Path, num_images: int) -> List[Path]:
    """Finds all image files in a directory and returns a random sample."""
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    all_files = [p for p in source_dir.iterdir() if p.suffix.lower() in image_extensions]
    
    if len(all_files) < num_images:
        print(f"Warning: Requested {num_images} images, but only found {len(all_files)}. Using all available images.")
        num_images = len(all_files)
        
    return random.sample(all_files, num_images)

def main():
    """Main function to run the augmentation and saving process."""
    parser = argparse.ArgumentParser(description="Visualize image augmentations from the ultralytics library.")
    parser.add_argument('--source-dir', type=str, required=True, help="Path to the directory with original images.")
    parser.add_argument('--output-dir', type=str, required=True, help="Path to the directory to save augmented images.")
    parser.add_argument('--num-images', type=int, default=10, help="Number of random images to process from the source directory.")
    parser.add_argument('--num-versions', type=int, default=5, help="Number of augmented versions to create per image.")
    args = parser.parse_args()

    source_path = Path(args.source_dir)
    output_path = Path(args.output_dir)

    if not source_path.is_dir():
        print(f"Error: Source directory not found at '{source_path}'")
        return
        
    output_path.mkdir(parents=True, exist_ok=True)

    # Define the augmentation pipeline 
    # We use RandAugment, a strong policy for good visualization.
    # The output is a PyTorch tensor with values in the range [0, 1].
    augment_transform = classify_augmentations(auto_augment="randaugment", imgsz=224)
    
    # Select Images
    image_paths = get_image_paths(source_path, args.num_images)
    if not image_paths:
        print("No images found in the source directory.")
        return

    # A transform to convert a tensor back to a PIL Image for saving
    tensor_to_pil = ToPILImage()

    print(f"Processing {len(image_paths)} images to create {args.num_versions} augmented versions of each...")

    # Process and save images
    for img_path in tqdm(image_paths, desc="Processing images"):
        try:
            original_image = Image.open(img_path).convert("RGB")
            
            # Create and save multiple augmented versions
            for i in range(args.num_versions):
                augmented_tensor = augment_transform(original_image)
                augmented_image = tensor_to_pil(augmented_tensor)
                
                # Define a new filename for the augmented image
                new_filename = f"{img_path.stem}_aug_{i}{img_path.suffix}"
                save_path = output_path / new_filename
                
                augmented_image.save(save_path)

        except Exception as e:
            print(f"Failed to process {img_path}: {e}")
            
    print(f"\nDone. Augmented images saved in '{output_path}'.")

if __name__ == "__main__":
    main()