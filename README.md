# Detectron2_Face_Mask_Detection

Detectron2 (official library Github) is “FAIR’s next-generation platform for object detection and segmentation”. FAIR (Facebook AI Research) created this framework to provide CUDA and PyTorch implementation of state-of-the-art neural network architectures. They also provide pre-trained models for object detection, instance segmentation, person keypoint detection and other usages.

The important, but often overlooked feature of the Detectron2 is its licensing scheme: the library itself is released under Apache 2.0 licence and pre-trained models are released under CC BY-SA 3.0 license. It means that you can modify the existing code, use it for private, scientific and even commercial purposes. All you need to do is provide appropriate credit to FAIR. It’s quite uncommon in scientific community, which often uses licenses forcing code source publication and non-commercial use. It’s quite limiting, but fortunately that’s not the case for Detectron2.

## Objective and Approach

* objective of this project is to predict if people are wearing mask or not. We have used detectron 2 object detection model ```COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml``` to train images and the output is preety accurate
* We trained the model for 1000 steps to gain near to better accuracy for the unseen data(images)
* All the configuration based on the requirements are set in the config file of the model
*

## Result of test data

![2021_3$largeimg_426024703 jpg_OD](https://user-images.githubusercontent.com/29440153/167271015-3953c34f-96ee-4027-993a-6b12d240dced.jpg)
![image1](https://user-images.githubusercontent.com/29440153/167271018-05d349a8-4fbb-46cd-9926-91eea1dae06f.jpg)


## Steps to train and predict in local machine

* Training can be done only when you have a graphic card in your system as the configuration is set to use the GPU from training
* CudaToolkit and CUDNN drivers are needed to be compatible with your pytorch version (please visit pytorch website to check for compatibility list)

* clone this repository in your local machine 
* Download the files from the given google drive link and paste it in your cloned folder
* ```pip install -r requirements.txt``` to install all dependent libraries ( Please check your pytorch version as it might cause the issue)
* run training.py to start training the data based on the dataset provided in google drive
* run prediction.py (for now please provided the test image path in the prediction class object to get a prediction file in your folder)


## On going Work

Creation of simple web app to remove the dependency of putting image name manually in the script

## Contributor
https://github.com/rohitp1995
