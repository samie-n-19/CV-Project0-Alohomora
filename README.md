# CV-Project0-Alohomora

_WPI RBE-549 Coursework | Computer Vision_

This repository contains my solutions for Project 0 ("Alohomora") as part of the RBE-549 Computer Vision course at Worcester Polytechnic Institute. The project is divided into two major phases, showcasing both classical and deep learning approaches for image analysis.

---

## Phase 1: Image Processing and Feature Extraction

### Objectives
- Implement various image processing techniques.
- Extract features from images using different filter banks.
- Generate and visualize texton maps, brightness maps, and color maps.
- Compute gradients using chi-square distance.

### Steps
1. **Generate Filter Banks**:
    - Difference of Gaussian (DoG) Filter Bank
    - Leung-Malik (LM) Filter Bank, LMS and LML
    - Gabor Filter Bank
    - Half-Disc Filter Bank

2. **Apply Filters**:
    - Apply the generated filter banks to the images.
    - Visualize and save the filter responses.

3. **Generate Maps**:
    - Texton Map
    - Brightness Map
    - Color Map

4. **Compute Gradients**:
    - Compute gradients using chi-square distance for texton, brightness, and color maps.

### Usage
To run the Phase 1 code, first go to this directory `cd Phase1/Code` and then execute the following command:
```sh
python3 Wrapper.py
```
## Phase 2: Deep Learning Models

### Objectives
- Implement and train deep learning models for image classification.
- Evaluate the performance of the models on the CIFAR-10 dataset.
- Apply data augmentation techniques to improve model generalization.

### Models Implemented
- CIFAR10Model
- ResNet18
- ResNeXt50
- DenseNet121

### Training all the networks
To train and run all four networks, execute the following commands:

Before running the network, always make sure that you uncomment the model name.

## Important Note
Before running the network, always make sure that you uncomment the model name in the `Train.py` script. Only one model should be uncommented at a time. Here is an example of how to set the model:

```python
# Uncomment the model you want to use
# model = CIFAR10Model(InputSize=3*32*32, OutputSize=10).to(device)
# model = ResNet18().to(device) 
model = ResNeXt50().to(device)
# model = DenseNet121().to(device)
```

#### Train the Network
```sh
python3 Train.py --CheckPointPath ../Checkpoints/ --NumEpochs 50 --DivTrain 1 --MiniBatchSize 32 --LoadCheckPoint 0 --LogsPath Logs/
```

### Testing all the networks
To test the trained models using the checkpoints, first go to the this directory `Phase2/Code` and then execute the following commands:

#### CIFAR10Model
```sh
python3 Test.py --ModelPath Checkpoints/CNN_network.ckpt
```

#### Custom CIFAR10Model
```sh
python3 Test.py --ModelPath Checkpoints/Custom_CNN_network.ckpt
```

#### ResNet18 Network
```sh
python3 Test.py --ModelPath Checkpoints/ResNet_Network.ckpt
```

#### ResNeXt50 Network
```sh
python3 Test.py --ModelPath Checkpoints/ResNeXt_Network_20.ckpt
```

#### DenseNet121 Network
```sh
python3 Test.py --ModelPath Checkpoints/DenseNet_network.ckpt
```

