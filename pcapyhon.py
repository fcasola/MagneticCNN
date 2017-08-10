# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 22:13:29 2017

@author: Francesco
"""

# principal value decomposition example
import numpy as np
import matplotlib.pyplot as plt
import struct
import sklearn
from sklearn.decomposition import PCA

root_dir = os.getcwd() #os.path.abspath('../..')
data_dir = os.path.join(root_dir, 'Tensorflow\\sub')
idx3 = os.path.join(data_dir, 't10k-images.idx3-ubyte')
idx1 = os.path.join(data_dir, 't10k-labels.idx1-ubyte')
 

with open(idx1, 'rb') as flbl:
    magic, num = struct.unpack(">II", flbl.read(8))
    lbl = np.fromfile(flbl, dtype=np.int8)
with open(idx3, 'rb') as fimg:
    magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))

    img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)
    
    threes = np.array([img[i] for i, l in enumerate(lbl) if l == 3])[:1000]
    
    plt.figure(10)
    plt.gray()
    plt.imshow(threes[2])

            
    model = PCA(n_components=300) ## this is a class performing PCA. n_components
    # represents the number of components you keep
    
    ## here is what I am doing right now:
        '''
        threes contains a collection of 3s
        the matrix is 1000x28x28. We reshape it in a 1000x784 matrix.
        Then we perform dimernionality reduction on the matrix, keeping 300 components only
        the 300 components having the largest eigenvalue!
        THIS CALCULATION IS BASED ON THE CORRELATION MATRIX AVERAGED OVER THE ENTIRE DATASET!!
        
        the result is in model.components_
        low is just the output:
        X_new : array-like, shape (n_samples, n_components)
        
        it contains the coefficients!! there are 300 coefficients for each dataset
        '''
    low = model.fit_transform(threes.reshape(1000, -1))
    
    ## result of the decomposition
    
    for i in range(5):
        plt.figure(i)
        plt.imshow(model.components_[i].reshape(28,28))


    # We are trying here to reconstruct the 1st three of the array three
    # we sum up progressively more and more components, trying to get the original number
    for i, comp in enumerate([5, 10, 50, 100, 300]):
        plt.figure(i)   
        plt.imshow((sum([low[1][i] * model.components_[i] for i in range(comp)])).reshape(28,28))    
    
    
    # there seems to be not much correlation between coefficients of different component vectors
    plt.figure(99)
    plt.scatter(low[:,0],low[:,99])
    
    
    
    
    
    
    
    
    
    
    
    
    
    