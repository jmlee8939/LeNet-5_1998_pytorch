# LeNet-5_1998_pytorch
by jaeminiman
LeNet-5 1988 version(pytorch)

## LeNet-5
LeNet-5 is proposed by Yann LeCun in 1988. This model is a pioneer of image recognition models using convolutional neural networks.
I want to reproduce this historical model as it was in 1998 with pytorch. The detailed description of LeNet-5_1998 is explained in [1]. 

## Structure


## Details to pay attention
There are some details of model which can be easy to missed. Because those are not used in the recent convolutional models.

### 1. activation function  
  - scaled hyperbolic tangent function
  - According to [1], LeNet-5 use scaled hyperbolic tangent function in order to prevent gradient vanishing problem.

### 2. partial connected convolutional layer
  - Feature maps in Layer C3 are not fully connected with all feature maps from S2.
  - In order to set the number of parameters to 60,000 (same with training dataset)
  
  
![activation_f](<./fig/activation_f.PNG>)
