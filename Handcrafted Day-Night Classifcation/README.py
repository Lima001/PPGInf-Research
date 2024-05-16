# Problem definition

Given a dataset of cropped vehicle images from various angles, the task is to accurately discern whether each image was captured during the day or at night.

# (Initial) Approach:

**Deep Learning:** Utilize direct training models like VGG-16, ResNet-18, and Inception V3 to extract features and classify images.

**Handcrafted Methods:** Investigate histogram thresholding techniques including HSV color channel thresholding and intensity thresholding for effective differentiation between daytime and nighttime images.

For this purpose, we are utilizing the following datasets: [BDD100K](https://openaccess.thecvf.com/content_CVPR_2020/html/Yu_BDD100K_A_Diverse_Driving_Dataset_for_Heterogeneous_Multitask_Learning_CVPR_2020_paper.html) (train and evaluation) and [VeRi-Wild](https://openaccess.thecvf.com/content_CVPR_2019/html/Lou_VERI-Wild_A_Large_Dataset_and_a_New_Method_for_Vehicle_CVPR_2019_paper.html) (evaluation).

To further challenge the classification problem, the methods are evaluated using cropped images of vehicles. This study aims to prompt the development of robust methods that rely solely on vehicle images, with limited scene information.

# Difficulties

Although the studied methods, especially the Deep Learning ones, may perform well on the training/evaluation dataset, the classifiers fail to perform adequately on evaluation-only data. Hence, the developed methods currently lack robustness.

# Notes

- **The code is under development; these are initial experiments**;
- Code is optimized to be executed on a single GPU;
- Datasets are not included;
- Datasets are expected to be formated following the ImageFolder (Pytorch) arrangement; 
- Deep Learning Models' parameters are not included;
- Deep Learning code is avaliable under folder "Image Classification"; the code was improved aiming generalization for arbitrary image classification task.
