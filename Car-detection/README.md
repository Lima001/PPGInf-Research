# About

Auxiliary code to detect and crop vehicle images from the BDD100K dataset. These images will be utilized for the Day/Night image classification task. It's worth mentioning that the code is generalized to perform vehicle detection on an entire directory; thus, to process a dataset, multiple calls need to be done.

# Specifications
- YOLOv8 is the default model utilized for the aforementioned task.
- For nighttime images, the minimum confidence threshold is expected to be lower compared to daytime images (values used are 0.7 and 0.8, respectively). This adjustment accounts for the slight difficulty, as experimentally visualized, in detecting vehicles in low-light conditions.
