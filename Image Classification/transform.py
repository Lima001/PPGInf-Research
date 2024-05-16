import numpy as np
from PIL import Image

class Transform():
    
    def __init__(self,transform):
        self.transform=transform
    
    def __call__(self,image):
        return self.transform(image=image)["image"]
    
    @staticmethod
    def open_img(img_path):
        img = Image.open(img_path)
        return np.array(img)