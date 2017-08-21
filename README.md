# Magnetic_CNN
Convolutional Neural Nets for magnetism problems.

## Aim
This code tests whether CNNs can "learn" the laws of magnetism
and are capable of reconstructing magnetization patterns given the field 
that they produce.

This code is inspired by recent publications addressing the possibility 
of machine learning algorithms to learn the laws of physics, e.g.:

https://arxiv.org/pdf/1607.03597.pdf

For the particular case of magnetism, we start from the formalism 
derived by us in a recent publication at Harvard:

https://arxiv.org/abs/1611.00673

In this first version, the magnetization is assumed to be collinear but
its orientation and spatial distribution is not known and will be the target 
of the learning process.

## Model
![Alt text](readme_img/Scheme_cnn.png?raw=true "model for the convolutional net implemented")

### Prerequisites
```
Written for python 3
Modules:
-scipy
-h5py
-argparse
-progressbar
-tensorflow
```

### Installing
```
git clone https://github.com/fcasola/MagneticCNN 
cd MagneticCNN/bin
python magnetic_cnn
```

### Configuration File
```
The configuration file is in data/Config/config_training.cfg. 
You can change it as per your need. Here is what the default 
configuration file looks like:

[Dataset]
Ntrain_ex: 1000
img_size: [32,32]
finer_grid: 4
rmax_v: 5

[model]
act_type: relu
k_filt_str_lyr1: [3,3,1]
Depth_lyr1: 16
Pool1_str_lyr1: [2,2]
Pool2_str_lyr1: [2,2]
k_filt_str_lyr2: [3,3,1]
Depth_lyr2: 16
k_filt_str_lyr3: [3,3,1]
Depth_lyr3: 16

[training]
learn_rate: 0.05
epochs: 30
batch_size: 2 

[paths]
save_train_set: ..\data\Training
save_learned: ..\data\Learned
```

### About the modules
```
Create_TrainingCNN.py 
- Produces the training set generating 2D magnetization patterns,
chosen to be ellipses of random axes and tilt. The module contains 
the function Compute_Bz, capable of finite element computation of 
the stray field given a certain magnetization pattern.

config.py 
- Import the configuration file

Learning_cnn.py
- Performs training over the generated dataset
```



