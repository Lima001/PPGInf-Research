# This script applies a set of image transformations to all images in a specified input directory
# and saves the transformed images to an output directory. The transformations include resizing,
# affine transformations, brightness and contrast adjustments, blurring, and dropout.
# Nonetheless, you can define other transformations and customize as necessary. 

import argparse
import os
import cv2
from albumentations import *
import numpy as np

def apply_transformations(image):
    """
        Apply a set of transformations to the given image.

        Args:
            image (numpy array): the input image.

        Returns:
            The transformed image (numpy array).
    """
    
    # Define the transformation that will be applied
    transform = Compose([
        Resize(224,224),
        Affine(scale=(0.9,1.2), rotate=(-180,180), shear=(-25,25), translate=(0.0,0.25), p=0.5),
        RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        GaussianBlur(p=0.4),
        CoarseDropout(max_holes=1, max_height=86, max_width=86, min_holes=1, min_height=86, min_width=86, fill_value="random", p=0.3),
    ])

    # Apply the transformation and return the result
    transformed = transform(image=image)
    return transformed['image']

def process_images(input_dir, output_dir):
    """
        Apply transformations to all images in an input directory and save them in an output directory.

        Args:
        - input_dir: Directory containing the input images.
        - output_dir: Directory where the transformed images will be saved.
    """
    
    # Create output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Process each image file in the input directory
    for filename in os.listdir(input_dir):
        
        input_image_path = os.path.join(input_dir, filename)
        
        if os.path.isfile(input_image_path) and input_image_path.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif')):
            
            # Read the image
            image = cv2.imread(input_image_path)
            
            # Convert image from BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Apply transformations
            transformed_image = apply_transformations(image)
            
            # Convert image back from RGB to BGR
            transformed_image = cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR)
            
            # Save the transformed image
            output_image_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_image_path, transformed_image)

            print(f"Transformed image saved to {output_image_path}")

def main():
    
    # Argument parser setup
    parser = argparse.ArgumentParser(description='Apply Albumentations transformations to all images in a specified directory.')
    parser.add_argument('--input_dir', type=str, help='Directory containing images to transform')
    parser.add_argument('--output_dir', type=str, default='transformed', help='Directory to save transformed images')

    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir

    # Process images
    process_images(input_dir, output_dir)

if __name__ == "__main__":
    main()

