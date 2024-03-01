# Inspired by:
#   K. H. Park and Y. S. Lee
#   Classification of Daytime and Night Based on Intensity and Chromaticity in RGB Color Image
#   2018 International Conference on Platform Technology and Service (PlatCon) 
#   Jeju, Korea (South), 2018, pp. 1-6, doi: 10.1109/PlatCon.2018.8472764
# classification method
# Method description: Day-Night Classifier based on Intensity in RGB Color Image

# Obs. Chromacity and K-means segementation - presented in the paper - were not used!

import sys
import cv2
import csv
import numpy as np

# Base threshold; adapted from the above-mentioned reference to consider only day-night time classification
BASE_NIGHT_DAY = 65

def intensity_classifier(image, threshold):
    
    # Get normalized intensity image
    intensity_image = np.mean(image, axis=-1)

    # Calculate the quantity of darker and brighter pixels based on intensity
    d_pixels = cv2.threshold(intensity_image,threshold,1,cv2.THRESH_BINARY)[1].sum()
    n_pixels = cv2.threshold(intensity_image,threshold,1,cv2.THRESH_BINARY_INV)[1].sum()

    # daytime image (label=0)
    return n_pixels > d_pixels

# Iterate over the training images, perform classification and return accuracy obtained
# To classify the validation dataset pass val_root_dir and val_label_file parameters
def classify_dataset(thresholding, train_root_dir, train_label_file, val_root_dir=None, val_label_file=None):

    corrects_train = 0
    n_inputs_train = 0

    with open(train_label_file) as file:
        csv_reader = csv.reader(file, delimiter=',')
        
        # row := [image_filename,label]
        for row in csv_reader:
            
            image = cv2.imread(f"{train_root_dir}/{row[0]}")
            predict = intensity_classifier(image, thresholding)            
            corrects_train += predict == int(row[1])
            n_inputs_train += 1

    if val_root_dir is not None and val_label_file is not None:

        corrects_val = 0
        n_inputs_val = 0

        with open(val_label_file) as file:
            csv_reader = csv.reader(file, delimiter=',')
            
            # row := [image_filename,label]
            for row in csv_reader:
                
                image = cv2.imread(f"{val_root_dir}/{row[0]}")
                predict = intensity_classifier(image, thresholding)
                corrects_val += predict == int(row[1])
                n_inputs_val += 1


        return (corrects_train/n_inputs_train, corrects_val/n_inputs_val)

    # Return single-element tuple (keep return pattern)
    return (corrects_train/n_inputs_train, )

if __name__ == "__main__":
    
    # Receive classify_dataset function parameters from terminal args
    train_root_dir = sys.argv[1]
    train_label_file = sys.argv[2]
    val_root_dir = None
    val_label_file = None

    # Expected validation dataset info
    if len(sys.argv) == 5:
        val_root_dir = sys.argv[3] 
        val_label_file = sys.argv[4]

    thresholding = BASE_NIGHT_DAY
    acc1 = classify_dataset(thresholding, train_root_dir, train_label_file, val_root_dir, val_label_file)
    acc2 = classify_dataset(thresholding*0.8, train_root_dir, train_label_file, val_root_dir, val_label_file)

    print(acc1,acc2,sep='\n')