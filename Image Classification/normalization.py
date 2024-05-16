import cv2
import argparse
import numpy as np

def get_norm_params(reference_file):
    
    with open(reference_file, 'r') as f:

        lines = f.readlines()

        mean = np.array([0.,0.,0.])
        stdTemp = np.array([0.,0.,0.])
        std = np.array([0.,0.,0.])

        numSamples = len(lines)

        for i in range(numSamples):
            im = cv2.imread(lines[i][:-1])
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im = im.astype(float) / 255.
        
            for j in range(3):
                mean[j] += np.mean(im[:,:,j])

        mean = (mean/numSamples)

        for i in range(numSamples):
            im = cv2.imread(lines[i][:-1])
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im = im.astype(float) / 255.
            for j in range(3):
                stdTemp[j] += ((im[:,:,j] - mean[j])**2).sum()/(im.shape[0]*im.shape[1])

        std = np.sqrt(stdTemp/numSamples)

        return mean, std
        

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_file")
    args = parser.parse_args()

    mean, std = get_norm_params(args.ref_file)
    print(f"mean:\n{mean}", f"std:\n{std}")