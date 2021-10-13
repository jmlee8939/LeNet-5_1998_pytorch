# LeNet-5_1998_pytorch
LeNet-5 1988 version by pytorch

# LeNet-5
LeNet-5 is proposed by Yann LeCun in 1988. This model is a pioneer of image recognition models using convolutional neural networks.
I want to reproduce this historical model as it was in 1998 with pytorch. The detailed description of LeNet-5_1998 is explained in [1]. 

## Details to pay attention
There are some details of model which can be easy to missed. Because those are not used in the recent convolutional models.

1. activation function  
  - According to [1], LeNet-5 use modified hyperbolic tangent function.
