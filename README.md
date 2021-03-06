# LeNet-5_1998_pytorch
by jaeminiman  
LeNet-5 1988 version(pytorch)

## LeNet-5
LeNet-5 is proposed by Yann LeCun in 1988. This model is a pioneer of image recognition models using convolutional neural networks.
I want to reproduce this historical model as it was in 1998 with pytorch. The detailed description of LeNet-5_1998 is explained in [[1]](https://ieeexplore.ieee.org/abstract/document/726791) 

## Structure
![structure](<./fig/structure.PNG>)

## Details to pay attention
There are some details of model which can be easy to missed. Because those are not used in the recent convolutional models.

### 1. activation function  
  - scaled hyperbolic tangent function
  - According to [[1]](https://ieeexplore.ieee.org/abstract/document/726791) 
, LeNet-5 use scaled hyperbolic tangent function in order to prevent gradient vanishing problem.  


![activation_f](<./fig/activation_f.PNG>)

### 2. partial connected convolutional layer
  - Feature maps in Layer C3 are not fully connected with all feature maps from S2.
  - In order to set the number of parameters to 60,000 (same with training dataset)    


![feat_connection](<./fig/feature map connection.PNG>)

### 3. RBF kernel
  - output layer is composed of Euclidean radial basis function units(RBF)
  - fixed parameter vectors(+1 or -1 only) from stylized image of the corresponding character class


![RBF kernel](<./fig/RBF kernel.PNG>)

### 4. Loss function
  - MSE + penalties of the incorrect classes


![loss f](<./fig/loss_f.PNG>)

# Reference
[1]
@article{lecun1998gradient,
  title={Gradient-based learning applied to document recognition},
  author={LeCun, Yann and Bottou, L{\'e}on and Bengio, Yoshua and Haffner, Patrick},
  journal={Proceedings of the IEEE},
  volume={86},
  number={11},
  pages={2278--2324},
  year={1998},
  publisher={Ieee}
}
