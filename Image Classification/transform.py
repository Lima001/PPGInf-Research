# This module provides a simple interface for applying image transformations using a transform object.
# It includes a class for handling transformations, which can be applied to images in a consistent manner.
# Additionally, it provides a method to open an image file and convert it into a NumPy array.

import numpy as np
from PIL import Image

class Transform():
    
    def __init__(self,transform):
        """
            Initializes the Transform object with the specified transformation.
            
            Args:
                transform: A transformation object that will be applied to images.
        """
        
        self.transform=transform
    
    def __call__(self,image):
        """
            Applies the stored transformation to the given image.
            
            Args:
                image: The input image to be transformed.
            
            Returns:
                Transformed image.
        """
        return self.transform(image=image)["image"]
    
    @staticmethod
    def open_img(img_path):
        """
            Opens an image file and converts it into a NumPy array.
            
            Args:
                img_path (str): The file path to the image.
            
            Returns:
                np.array: Image as a NumPy array.
        """
        img = Image.open(img_path)
        return np.array(img)