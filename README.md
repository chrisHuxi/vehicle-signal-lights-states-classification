[![Project from BMW](https://github.com/chrisHuxi/Trajectory_Predictor/blob/master/readme_images/Absolut.jpeg)](https://absolut-projekt.de/)

[![Project from TUD](https://img.shields.io/badge/TU%20dresden-Computer%20Science-blue)](https://tu-dresden.de/ing/informatik)


# vehicle signal lights states classification
TUD master thesis project, vehicle signal lights states classification based on video clips, pytorch.

## Intro:
This thesis aims to solve the state classification problem of daytime vehicle signal rearlights based on images, which is an important module in an autonomous driving system. Based on previous methods, we propose and implement a novel model called "ResNet-LSTM network" to complete this classification task. Further, we try to merge the light region's position information into the current model and propose the "YOLO-ResNet-LSTM network",  which however decreases the performance and still has space for improvement. Besides the performance, we try to implement the model with TensorRT to achieve higher inference speed and less model size to ensure its real-time capability and efficiency. Finally, a model with outstanding results in both performance and efficiency is obtained, the accuracy of the classification can reach 94.9\% on 8-classes rearlight state classification tasks, and the average inference speed on GPU can reach 36.1 ms per video clip of 10 frames.

 ## Proposed model
 
 ### ResNet-LSTM network
 The ResNet-LSTM got a impressive result, the architecture is shown:
 
 
 <div align=center> <img src="https://github.com/chrisHuxi/vehicle-signal-lights-states-classification/blob/master/readme/evluation/proposed_model/ResNet-LSTM.png" alt="drawing" width="500"/> </div>


 ### YOLO-ResNet-LSTM network
 The YOLO-ResNet-LSTM however got a worse result, the architecture is shown:
 
 
 <div align=center> <img src="https://github.com/chrisHuxi/vehicle-signal-lights-states-classification/blob/master/readme/evluation/proposed_model/YOLO-ResNet-LSTM.png" alt="drawing" width="500"/> </div>
  
## Usage:

### Config enviroment:
  ```bash
  python3 -m venv /path/to/your/virtual/env
  source /path/to/your/virtual/env
  pip install requirements.txt
  ```

### Train model:

#### ResNet-LSTM model:
The most important code file are the dataloader/VSLdataset.py, models/CNN_LSTM_model_resnet50.py and models/CNN_LSTM_model_infer_RT.py.
And you have to make sure your dataset structure looks like this:
```bash
train/
    - BOO/
        -clip1/
            -frame00.png
            -frame01.png
            - ...
        -clip2
            - frame00.png
            - frame01.png
            - ...
    - BLO
        -clip3/
            -frame00.png
            -frame01.png
            - ...
        -clip4
            - frame00.png
            - frame01.png
    - ... 
  valid/
      - BOO/
      -...
  test/
      - BOO/
      -...   
  ```
  
  Then you can start training by run script:
  ```bash
  python models/CNN_LSTM_model_resnet50.py
  ```
  where mainly includes 2 step: 1. create model 2. training model
  ```python
  model = CLSTM(lstm_hidden_dim = 512, lstm_num_layers = 3, class_num=8)        
  train(model_in = model, num_epochs = 100, load_model = False, freeze_extractor = False)
  ```
  For comparing different encoder network, decoder network and video clips of different length you can check the code in detail and modify the corresponding code.
  The model with best performance is the ResNet50-LSTM3 model, the trained model path is here: [**link??**]()
  
  To infer the model with data without labels, you can put all your images into a dir called "inference_data" as structrue:
  ```bash
  inference_data/
    - clip1
        -frame00.png
        -frame01.png
        - ...
    - clip2
        -frame00.png
        -frame01.png
        - ...
    - ... 
  ```
  
  Then you can start inference by comment the traning code and uncomment the inference code:
  ```python
  model = CLSTM(lstm_hidden_dim = 512, lstm_num_layers = 3, class_num=8)      
  infer(model)
  ```
  or if your want to run inference with tensorrt (you have to install [**torch2trt**](https://github.com/NVIDIA-AI-IOT/torch2trt) before. Besides, the original torch2trt dosen't have LSTM implementation on TRT, so you have to write the corresponding converter as [**link**](https://github.com/NVIDIA-AI-IOT/torch2trt/issues/144))
  ```bash
  python models/CNN_LSTM_model_infer_RT.py
  ```
#### YOLO-ResNet-LSTM model:
  To run YOLO network to detect the light before feeding into ResNet-LSTM:
  You have to install YOLO as instruction: [**link**](https://github.com/AlexeyAB/darknet)
  Then you can use the code /YOLO_models/[**??**]() to get the bbox of lights, then either cut out the ROI or feed as masks.
  The pre-trained model: [**link??**]().
  And you can create a new dataset folder named "YOLO_mask_dataset" which has the same structure of train/valid/test
  Then you can start training by run script:
  ```bash
  python models/CNN_LSTM_model_mask.py
  ```
  However, the result of this model is worse, which means the YOLO network misleads the classifier.
  
## Result:

### Statics:

<div align=center> <img src="https://github.com/chrisHuxi/vehicle-signal-lights-states-classification/blob/master/readme/table.PNG" alt="drawing" width="700"/> </div>

We choose a model with best performance: ResNet50-LSTM3 and find that when a longer sequence is applied only for the test phase while in the training phase the short sequence is applied, the classification result could be even better.

<div align=center> <img src="https://github.com/chrisHuxi/vehicle-signal-lights-states-classification/blob/master/readme/table_long.PNG" alt="drawing" width="700"/> </div>


### Visualized result:


#### Confusion matrix:

left figure is the confusion matrix of model inferring with 10 frames while right figure is with 20 frames.


<div align=center> <img src="https://github.com/chrisHuxi/vehicle-signal-lights-states-classification/blob/master/readme/evluation/cofusion_matrix_todo.png" alt="drawing" width="700"/> </div>
 
 
 
#### ROC curve:
left figure is the roc curve of model inferring with 10 frames while right figure is with 20 frames.


<div align=center> <img src="https://github.com/chrisHuxi/vehicle-signal-lights-states-classification/blob/master/readme/evluation/ROC_len10_len20.png" alt="drawing" width="700"/> </div>


## Dataset:
To train a neural network, a significant role is the dataset. For vehicle signal lights states classification, we have two types of datasets to use, which we call the "end-to-end method dataset" for directly classifying states from frames and the "detection-based method dataset" for detecting the lights' bounding box first.

### The end-to-end method dataset
We used a public dataset: [**link**](), some examples of each class is shown:


<div align=center> <img src="https://github.com/chrisHuxi/vehicle-signal-lights-states-classification/blob/master/readme/evluation/dataset.png" alt="drawing" width="500"/> </div>


### The detection-based method dataset
We labeled 715 images with bbox of lights: [**link**](), some examples of each class is shown:


<div align=center> <img src="https://github.com/chrisHuxi/vehicle-signal-lights-states-classification/blob/master/readme/evluation/dataset_detection.png" alt="drawing" width="500"/> </div>



