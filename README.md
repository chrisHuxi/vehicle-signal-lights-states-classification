# vehicle-signal-lights-states-classification
[![Project from BMW](https://github.com/chrisHuxi/Trajectory_Predictor/blob/master/readme_images/Absolut.jpeg)](https://absolut-projekt.de/)

[![Project from TUD](https://img.shields.io/badge/TU%20dresden-Computer%20Science-blue)](https://tu-dresden.de/ing/informatik)


# Trajectory Predictor
TUD master thesis project, vehicle signal lights states classification based on video clips, pytorch.

## Intro:
This thesis aims to solve the state classification problem of daytime vehicle signal rearlights based on images, which is an important module in an autonomous driving system. Based on previous methods, we propose and implement a novel model called "ResNet-LSTM network" to complete this classification task. Further, we try to merge the light region's position information into the current model and propose the "YOLO-ResNet-LSTM network",  which however decreases the performance and still has space for improvement. Besides the performance, we try to implement the model with TensorRT to achieve higher inference speed and less model size to ensure its real-time capability and efficiency. Finally, a model with outstanding results in both performance and efficiency is obtained, the accuracy of the classification can reach 94.9\% on 8-classes rearlight state classification tasks, and the average inference speed on GPU can reach 36.1 ms per video clip of 10 frames.

## Dataset:
To train a neural network, a significant role is the dataset. For vehicle signal lights states classification, we have two types of datasets to use, which we call the "end-to-end method dataset" for directly classifying states from frames and the "detection-based method dataset" for detecting the lights' bounding box first.
