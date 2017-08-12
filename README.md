# MagneticCNN
Convolutional Neural Net for magnetism problems.

## Getting Started
Description of the code

### Prerequisites
```
Written for python 3
Modules:
-scipy
-h5py
-argparse
-progressbar

```

### Installing
```
git clone https://github.com/fcasola/MagneticCN 
cd MagneticCNN/bin
python magnetic_cnn
```

### Configuration File
```
The configuration file is in data/Config/config_training.cfg. 
You can change it as per your need. Here is what the default 
configuration file looks like:

[Dataset]
img_size: [32,32]
finer_grid: 4
rmax_v: 5

[paths]
save_train_set: ..\data\Training
```

### About the modules
```
Create_TrainingCNN.py - Produces the training set using the theory 
for magnetic field propagation derived in  https://arxiv.org/abs/1611.00673. 
The module contains the function Compute_Bz, capable of finite element 
computation of the stray field given a certain magnetization pattern.

config.py - Import the configuration file

```



