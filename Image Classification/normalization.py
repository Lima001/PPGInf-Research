# This script calculates the mean and standard deviation of pixel values for a set of images.
# The mean and standard deviation are computed channel-wise (for RGB channels) and can be used 
# for image normalization in deep learning tasks. The script reads the image paths from a reference file,
# processes each image, and computes the statistics across all the images.

import cv2
import argparse
import numpy as np

def get_norm_params(reference_file):
    """
        Calculates the mean and standard deviation for the RGB channels of the images listed in the reference file.

        Args:
            reference_file (str): Path to a text file containing the paths to the images.
        
        Returns:
            mean (np.array): The mean pixel values for the R, G, and B channels.
            std (np.array): The standard deviation of pixel values for the R, G, and B channels.
    """
    
    with open(reference_file, 'r') as f:

        lines = f.readlines()

        mean = np.array([0., 0., 0.])       # Initialize mean array for R, G, B channels
        stdTemp = np.array([0., 0., 0.])    # Temporary array to accumulate squared differences
        std = np.array([0., 0., 0.])        # Final standard deviation array
        numSamples = len(lines)             # Total number of images

        # Calculate mean for each channel
        for i in range(numSamples):
            im = cv2.imread(lines[i][:-1])              # Read the image
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)    # Convert from BGR to RGB
            im = im.astype(float) / 255.                # Normalize pixel values to [0, 1]
        
            # Iterate over R, G, B channels and accumulate mean of each channel
            for j in range(3):                          
                mean[j] += np.mean(im[:,:,j])

        # Average the accumulated means
        mean = (mean/numSamples)

        # Calculate standard deviation for each channel
        for i in range(numSamples):
            im = cv2.imread(lines[i][:-1])
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im = im.astype(float) / 255.
            
            # Iterate over R, G, B channels and accumulate squared differences
            for j in range(3):
                stdTemp[j] += ((im[:,:,j] - mean[j])**2).sum()/(im.shape[0]*im.shape[1])

        # Compute the standard deviation by taking the square root of the mean of squared differences
        std = np.sqrt(stdTemp/numSamples)

        return mean, std
        

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_file")       # Path to the reference file containing image paths (one by line) that will compute the mean and std
    args = parser.parse_args()

    mean, std = get_norm_params(args.ref_file)
    print(f"mean:\n{mean}", f"std:\n{std}")