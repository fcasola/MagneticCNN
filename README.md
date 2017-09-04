# Magnetic_CNN
Convolutional Neural Nets for magnetism problems.

## Aim
Computing the magnetic field produced by a certain magnetization pattern is a task performed by many available [numerical routines](http://math.nist.gov/oommf/), with applications ranging from science to engineering. What about the opposite calculation instead? Is it possible, in general, to reconstruct a certain magnetization pattern given the field it produces?

We have recently been the first to answer this question and studied the related math in one of our publications at Harvard:

https://arxiv.org/abs/1611.00673

Turns out that obtaining the manifold of solutions in the general case is an extremely non-trivial task, involving several complex numerical variational minimizations.

Here we test the possibility of having a general, single piece of code capable of performing this task making use of the power of convolutional neural networks ([CNNs](https://en.wikipedia.org/wiki/Convolutional_neural_network)). 
We test the capability of CNNs of "learning" the laws of magnetism and perform for us all the related image processing operations.

This code is inspired by recent publications addressing the possibility of machine learning algorithms to learn other, extremely important for applications, laws of physics, e.g.:

https://arxiv.org/pdf/1607.03597.pdf

In this first version, the code is capable of reconstructing simple collinear magnetization patterns of arbitrary shape and orientation. 
In the next versions, we will teach our CNN to recognize more complex textures such as spirals, cycloids and all the other structures belonging [to the realm of classical magnetism.](https://arxiv.org/abs/1611.00673) 

## Model

![Alt text](readme_img/training_1.png?raw=true "model for the convolutional net implemented")

In the present model, we start generating Ns randomly defined magnetization patterns. The magnetization is collinear and has an elliptical-like shape (see exemplary image above, showing 3 shapes defined on a NpxNp = 32x32 pixel grid. The magnetization direction in space (parametrized by polar and azimuthal angle q,p) is sampled uniformly over the sphere. 

Next, for each magnetization we compute the out-of-plane component of the magnetic field using the "Compute_Bz" function in the "Create_TrainingCNN" module. Data are then saved in a hierarchical data format (.h5) file as [q,p,m,Bz] with dimensions [Ns x (2 + 2* Np**2 )]. The (m,Bz) pair is then used as the (y,x) label/feature of the cnn.

Importantly, we use dimensional scaling in the images to facilitate the learning process!
 
![Alt text](readme_img/Scheme_cnn.png?raw=true "model for the convolutional net implemented")

The .h5 dataset is then fed for training into a cnn mimicking the one in https://arxiv.org/pdf/1607.03597.pdf. The scheme for the cnn is in the figure above. At prediction time, the user can feed the network using the module "Predict_cnn" using a lossless .png picture containing the out-of-plane component of the magnetic field. The png is created such that b/w is the minimum/maximum (Min/Max) value for the magnetic field. Predict_cnn needs to know only the (Min/Max) values and the range of the spatial axes (width/height of the image, see python Predict_cnn.py -h for detailed help).

### Prerequisites

- Python 3.6
- [Tensorflow 1.2.1](https://pypi.python.org/pypi/tensorflow/1.2.1)
- [SciPy](http://www.scipy.org/install.html)
- [h5py](http://www.h5py.org/)
- [Progressbar](https://pypi.python.org/pypi/progressbar2/3.18.1)
- [argparse](https://docs.python.org/3/library/argparse.html)

### Installing
First, do:

```
git clone https://github.com/fcasola/MagneticCNN 
```

To facilitate use, a test dataset and learning parameters are already included in this github version.
To train the CNN yourself, run the following command:

```
cd MagneticCNN/bin
python magnetic_cnn
```

The CNN will be trained according to the parameters contained in the configuration file data/Config/config_training.cfg.

To run predictions, prepare a .png image with the stray field, place it in the folder MagneticCNN\data\Predictions and follow the help of:

```
cd MagneticCNN/magn_CNN
python Predict_cnn.py -h
```

A simple example can run by typing:

```
cd MagneticCNN/magn_CNN
python Predict_cnn.py -e
```

The example will be reconstructing an elliptical shape for the magnetization as defined in data/Config/test_shape.cfg. You can change the shape and the magnetization orientation yourself and have fun with the CNN!!
The result for the default case is shown in the section 'Results' below.

All predictions will be saved in the folder data/Predictions

### A few words about the modules
```
# Create_TrainingCNN.py 
Produces the training set generating 2D magnetization patterns,
chosen to be ellipses of random axes and tilt. The module contains 
the function Compute_Bz, capable of finite element computation of 
the stray field given a certain magnetization pattern.

usage: Create_TrainingCNN.py [-h] [-o] [-n]
Create training data for the MCNN.

optional arguments:
  -h, --help      show this help message and exit
  -o , --Output   Filename for the hdf5 output
  -n , --Ndata    Number of training samples (integer)

# config.py 
- Import the configuration file

# Learning_cnn.py

usage: Learning_cnn.py [-h] [-o]
Training a CNN for magnetic problems.
optional arguments:
  -h, --help      show this help message
  -o, --Output    Filename prefix for saving tensorflow output 

# Predict_cnn.py

usage: Predict_cnn.py [-h] [-e] [-i] [-r ] [-c ] [-d] [-t] [-ms]
    Making predictions based on the trained cnn.
    The module takes as input an 8-bit pixels, black and white png image
    representing the stray field and creates an image with the same resolution
    representing the estimate of the cnn for a spatially varying collinear magnetization.
    Notes:
    1 - In using this function (see parameter definitions below),
    remember that the current model holds in the limit t<<d.
    2 - The b/w image containing the stray field is defined such that
    b/w (0-255) is the min/max of the field in Gauss.
    3 - The output is saved with similar conventions as in 2), where
    now b/w is 0/1 in Ms units. Look at "*_predicted.png"

optional arguments:
  -h, --help          show this help message and exit
  -e, --example       Creates exemplary Bz data/label for the cnn using a test shape specified in the ''Config'' folder.
  -i , --Image        Image name for the stray field.
                      The Image must be in the ''Predictions'' folder.
  -r  , --XYrange     Spcify width and height of the stray field image [in meters].
  -c  , --Caxrange    Spcify minimum and maximum value of the field [in Gauss].
  -d , --Distance     Distance of the stray field to the magnetization plane [in meters].
  -t , --thickness    Thickness of the magnetic layer [in meters].
  -ms , --Ms          Maximum scalar value for the nominal saturation magnetization of
                      the layer [in A/m].
```
## Results

Training the CNN took approximately 45 minutes on a NVIDIA Tesla K20Xm GPU.

![Alt text](readme_img/results_1.png?raw=true "Target magnetization (left). Stray field input to the CNN (center) and interpolated output using the current training (right).")

Target default magnetization defined in data/Config/test_shape.cfg (left). Stray field input to the CNN (center) and interpolated output using the current training (right)

![Alt text](readme_img/results_2.png?raw=true "Loss function vs epochs time using the config file parameters specified above.")

Loss function vs epochs time using the config file parameters specified above. Training data of 3000 images with elliptical shapes of random orientation, posiiton, size and tilt. Loss on the validation set (2000 images) at the final step is 9.55*10-4.



