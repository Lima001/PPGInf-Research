# Based on :
#   Saha, B.; Davies, D.; Raghavan, A. 
#   Day Night Classification of Images Using Thresholding on HSV Histogram. 
#   U.S. Patent 9,530,056, 27 Dezembro 2013; 
# classification method
# Method description: Day-Night Classifier based on HSV histogram thresholding

import sys
import cv2
import csv
import os
import numpy as np

# Baseline values normalized to [0,1] range to calculate Nh and Nv values (see above-mentioned reference for more details)
# Explanation:
#   Nh = Pixels with normalized hue in range [0, NORMALIZED_BASE_HUE_LOWER] U [NORMALIZED_BASE_HUE_UPPER, 1] 
#   Nv = Pixels with normalized value in range [NORMALIZED_BASE_VALUE, 1] 
NORMALIZED_BASE_HUE_UPPER = 288/360
NORMALIZED_BASE_HUE_LOWER = 72/360
NORMALIZED_BASE_VALUE = 150/255

# Get average ratio HUE and ratio VALEU of daytime images in a given train dataset
# The obtained value is further used as thresholding in image processing
def get_avg_thresholds(train_root_dir, train_label_file):
    
    hue_sum = 0
    value_sum = 0
    n_inputs = 0

    with open(train_label_file) as file:
        csv_reader = csv.reader(file, delimiter=',')
        
        # row := [image_filename,label]
        for row in csv_reader:
                
            # Verify if the given dataset entry is some daytime image (label=0)
            if int(row[1]) == 0:
                
                # Compute and normalize HSV channel values
                image = cv2.imread(f"{train_root_dir}/{row[0]}")
                image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                
                # Normalization
                image_hsv = np.float32(image_hsv)
                h,s,v = cv2.split(image_hsv)

                h /= 179.0
                v /= 255.0

                # Calculete Nh and Nv values (see the method reference for more details)
                nh_partial1 = cv2.threshold(h,NORMALIZED_BASE_HUE_UPPER,1,cv2.THRESH_BINARY)[1].sum()
                nh_partial2 = cv2.threshold(h,NORMALIZED_BASE_HUE_LOWER,1,cv2.THRESH_BINARY_INV)[1].sum()
                nh = nh_partial1 + nh_partial2
                nv = cv2.threshold(v,NORMALIZED_BASE_VALUE,1,cv2.THRESH_BINARY)[1].sum()
                
                # Calculete Hue and Value ratio in the processed image
                hue_sum += nh/(image.shape[0]*image.shape[1])
                value_sum += nv/(image.shape[0]*image.shape[1])

                n_inputs += 1


    # Return the average ratio (for Hue and Value channel) in the given daytime images 
    return (hue_sum/n_inputs, value_sum/n_inputs)

def classifier(image, thresholding_h, thresholding_v):

    # Pre-process image for classification; map to HSV color space and normalize to range [0,1]
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image_hsv = np.float32(image_hsv)
    h,s,v = cv2.split(image_hsv)

    h /= 179.0
    v /= 255.0

    # Calculete Nh and Nv values (see the method reference for more details)
    nh_partial1 = cv2.threshold(h,NORMALIZED_BASE_HUE_UPPER,1,cv2.THRESH_BINARY)[1].sum()
    nh_partial2 = cv2.threshold(h,NORMALIZED_BASE_HUE_LOWER,1,cv2.THRESH_BINARY_INV)[1].sum()
    
    nh = (nh_partial1 + nh_partial2)/(image.shape[0]*image.shape[1])
    nv = cv2.threshold(v,NORMALIZED_BASE_VALUE,1,cv2.THRESH_BINARY)[1].sum()/(image.shape[0]*image.shape[1])

    return int(nh > thresholding_h and nv < thresholding_v)


# Iterate over the training images, perform classification and return accuracy obtained
# To classify the validation dataset pass val_root_dir and val_label_file parameters
def classify_dataset(thresholding_h, thresholding_v, train_root_dir, train_label_file, val_root_dir=None, val_label_file=None):

    corrects_train = 0
    n_inputs_train = 0

    with open(train_label_file) as file:
        csv_reader = csv.reader(file, delimiter=',')
        
        for row in csv_reader:

            
            image = cv2.imread(f"{train_root_dir}/{row[0]}")
            predict = classifier(image, thresholding_h, thresholding_v)
            corrects_train += (predict == int(row[1]))
            n_inputs_train += 1

    if val_root_dir is not None and val_label_file is not None:

        corrects_val = 0
        n_inputs_val = 0

        with open(val_label_file) as file:
            csv_reader = csv.reader(file, delimiter=',')
            
            for row in csv_reader:

                image = cv2.imread(f"{val_root_dir}/{row[0]}")
                predict = classifier(image, thresholding_h, thresholding_v)
                corrects_val += (predict == int(row[1]))
                n_inputs_val += 1

        return (corrects_train/n_inputs_train, corrects_val/n_inputs_val)

    # Return single-element tuple (keep return pattern)
    return (corrects_train/n_inputs_train, )

def perform_inference(thresholding_h, thresholding_v, root_dir):

    day_count = 0
    night_count = 0

    for filename in os.listdir(root_dir):
        
        if os.path.isfile(f"{root_dir}/{filename}"):            
        
            image = cv2.imread(f"{root_dir}/{filename}")
            predict = classifier(image, thresholding_h, thresholding_v)
        
            day_count += predict == 0
            night_count += predict == 1
                
    return (day_count, night_count)

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
    
    thresholding_h, thresholding_v = get_avg_thresholds(train_root_dir, train_label_file)
    #print(thresholding_h, thresholding_v) 0.31502976440243524 0.2401562484890622
    
    acc1 = classify_dataset(thresholding_h, thresholding_v, train_root_dir, train_label_file, val_root_dir, val_label_file)
    acc2 = classify_dataset(thresholding_h*0.8, thresholding_v*0.8, train_root_dir, train_label_file, val_root_dir, val_label_file)

    print(acc1,acc2,sep='\n')