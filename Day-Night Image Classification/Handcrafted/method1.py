# Inspired by https://github.com/jayeshsaita/Day-Night-Classifier base model classifier
# Method description: Day-Night Classifier based on brightness (value) thresholding on HSV color space

import sys
import cv2
import csv
import numpy as np

# Get average brightness of daytime images in a given train dataset
# The obtained value is further used as thresholding in image processing
def get_avg_brightness(train_root_dir, train_label_file):
    
    brightness_sum = 0.0
    n_inputs = 0

    with open(train_label_file) as file:
        csv_reader = csv.reader(file, delimiter=',')
        
        # row := [image_filename,label]
        for row in csv_reader:
            
            # Verify if the given dataset entry is some daytime image (label=0)
            if int(row[1]) == 0: 
            
                image = cv2.imread(f"{train_root_dir}/{row[0]}")
                image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                brightness_sum += np.mean(image_hsv[:, :, 2])
                n_inputs += 1

    return brightness_sum/n_inputs

# Classification method based on brightness thresholding
def classifier(image, thresholding):
    
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    brightness = np.mean(image_hsv[:, :, 2])
    
    # If brightness if less than the thresholding, then it's classified as nighttime image (label=1)
    return (brightness < thresholding)

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
            predict = classifier(image, thresholding)
            corrects_train += (predict == int(row[1]))
            n_inputs_train += 1

    if val_root_dir is not None and val_label_file is not None:

        corrects_val = 0
        n_inputs_val = 0

        with open(val_label_file) as file:
            csv_reader = csv.reader(file, delimiter=',')
            
            # row := [image_filename,label]
            for row in csv_reader:
                
                image = cv2.imread(f"{val_root_dir}/{row[0]}")
                predict = classifier(image, thresholding)
                corrects_val += (predict == int(row[1]))
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
    
    thresholding = get_avg_brightness(train_root_dir, train_label_file)
    
    acc1 = classify_dataset(thresholding, train_root_dir, train_label_file, val_root_dir, val_label_file)
    acc2 = classify_dataset(thresholding*0.8, train_root_dir, train_label_file, val_root_dir, val_label_file)
    
    print(acc1,acc2,sep='\n')