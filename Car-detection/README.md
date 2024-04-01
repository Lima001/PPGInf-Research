# About

Auxiliary code to detect and crop vehicle images from the BDD100K dataset. These images will be utilized for the Day/Night image classification task. It's worth mentioning that the code will be updated to be applicable to general datasets in the future and optimized to improve GPU usage.

# Specifications
-- YOLOv8 is the default model utilized for the aforementioned task.
-- For nighttime images, the minimum confidence threshold is set lower compared to daytime images. This adjustment accounts for the slight difficulty, as experimentally visualized, in detecting vehicles in low-light conditions.
