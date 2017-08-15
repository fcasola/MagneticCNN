"""
Convolutional Neural Network model for magnetism problems

written by F. Casola, Harvard University - fr.casola@gmail.com
"""

import numpy as np
import tensorflow as tf
import math as mt
from tensorflow.contrib import learn
# personal modules
from config import *


# define a function capable of upscaling a tensor
def upscaling(matr_in,factor,indim):
    """
    Recursive function performing upscaling in tensorflow.
    Dimensions of a general tensor [batch_size,width,height,channels]
    will be upscaled recursively such that the new dimension is
    [batch_size,width*2^factor,height*2^factor,channels]

    Input parameters are:
        *matr_in - matr_in, input matrix to upsample
        *factor - factor, scaled by 2^factor
        *indim - initial width and height dimensions of the tensor matr_in
        
    Returns
    ----------
        - the upscaled matrix matr_in 
    
    """
    if factor==0:
        return matr_in
    else:    
        a = matr_in
        b = tf.reshape(a, [-1,indim[0], indim[1], indim[2], 1])
        c = tf.tile(b, [1, 1, 1, 1, 2])
        d = tf.reshape(c, [-1,indim[0], indim[1]*2, indim[2]])        
        e = tf.reshape(d, [-1,indim[0], indim[1]*2, indim[2], 1])
        f = tf.tile(e, [1, 1, 1, 1, 2])
        g = tf.transpose(f, [0, 1, 4, 3, 2])
        h_fin = tf.reshape(g, [-1,indim[0]*2, indim[1]*2, indim[2]])
        return upscaling(h_fin,factor-1,[2*indim[0], 2*indim[1], indim[2]])

def cnn_model_fn(features, trg_map, mode):   
  """
    Model function for magnetic CNN.
    This function implements the convolutional neural net model 
    discussed in arxiv.org/pdf/1607.03597.pdf.
    Input parameters are:
        *features - input data; numpy float32 array with dimensions [batch_size,width*height]
        *trg_map - target output data; numpy float32 array with dimensions [batch_size,width*height]
        *mode - specifies the running mode. Definitions are
                according to the standard learn.ModeKeys keys.
                ** 'train': training mode.
                ** 'eval': evaluation mode.
                ** 'infer': inference mode.
    Returns
    ----------
        - the tuple (loss,optimizer), representing the type of loss function 
        and the optimizer method used.
  """
  # Reshape tensors to 4-D tensor: [batch_size, width, height, channels]
  # Magnetic field images are img_size[0]ximg_size[1] pixels, and have one channel
  input_layer = tf.reshape(features, [-1, img_size[0],img_size[1], 1])
  trg_map_rsh = tf.reshape(trg_map, [-1, img_size[0],img_size[1], 1])

  # Selecting activation type.    
  if act_type=='relu':
     act_type = tf.nn.relu
  else:
     act_type = tf.nn.sigmoid

  # LAYER 1 ---------------         
  # Convolutional layer # 1    
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=Depth_lyr1,
      kernel_size=k_filt_str_lyr1[0:2],
      padding="same",
      activation=act_type)  
  # Pooling1, layer 1
  pool1_lyr1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[Pool1_str_lyr1[0]]*2, 
                                       strides=Pool1_str_lyr1[1])
  # Pooling2, layer 1
  pool2_lyr1 = tf.layers.max_pooling2d(inputs=pool1_lyr1, pool_size=[Pool2_str_lyr1[0]]*2, 
                                  strides=Pool2_str_lyr1[1])
  
  # LAYER 2 ---------------         
  conv2 = tf.layers.conv2d(
      inputs=conv1,
      filters=Depth_lyr2,
      kernel_size=k_filt_str_lyr2[0:2],
      padding="same",
      activation=act_type)  
  # Pooling1, layer 2
  pool1_lyr2 = tf.layers.conv2d(
      inputs=pool1_lyr1,
      filters=Depth_lyr2,
      kernel_size=k_filt_str_lyr2[0:2],
      padding="same",
      activation=act_type)  
  # Pooling2, layer 2
  pool2_lyr2 = tf.layers.conv2d(
      inputs=pool2_lyr1,
      filters=Depth_lyr2,
      kernel_size=k_filt_str_lyr2[0:2],
      padding="same",
      activation=act_type)  
  
  # Do upscaling and combining
  # tensors conv2,pool1_lyr2 and pool2_lyr2 
  # Pooling MUST rescale the map by powers of 2 by design
  
  # Upscaling
  factor1 = mt.log(Pool1_str_lyr2[0],2)
  factor2 = mt.log(Pool2_str_lyr2[0]*Pool2_str_lyr1[0],2)
  if ((factor1%1)!=0 or (factor2%1)!=0)==True:
      raise ValueError('Pooling size/sizes not multiple of 2')        
  ndim1 = [int(img_size[0]/(2**factor1)),int(img_size[1]/(2**factor1)),Depth_lyr1] 
  ndim2 = [int(img_size[0]/(2**factor2)),int(img_size[1]/(2**factor2)),Depth_lyr1]
  US_pool1_lyr2 =  upscaling(pool1_lyr2,factor1,ndim1)
  US_pool2_lyr2 =  upscaling(pool2_lyr2,factor2,ndim2)

  # Combining Upscaled pooling layers with the regular Conv layer
  Combined_output = conv2 + US_pool1_lyr2 + US_pool2_lyr2
  
  # LAYER 3 ---------------         
  # Apply a 1x1 convolution to act with ReLu, as in https://arxiv.org/pdf/1607.03597.pdf
  conv3 = tf.layers.conv2d(
      inputs=Combined_output,
      filters=Depth_lyr2,
      kernel_size=k_filt_str_lyr2[0:2],
      padding="same",
      activation=act_type)   
  
  # LAYER 4 ---------------         
  # Apply a 1x1 convolution to reduce feature size
  conv4 = tf.layers.conv2d(
      inputs=conv3,
      filters=1,
      kernel_size=k_filt_str_lyr2[0:2],
      padding="same",
      activation=act_type)   
  
  # The loss function
  loss = None
  train_op = None
  
  if mode != learn.ModeKeys.INFER:  
      loss = tf.reduce_mean(tf.square(trg_map_rsh - conv4))
      
  if mode == learn.ModeKeys.TRAIN:
      # Adam optimizer as in https://arxiv.org/pdf/1607.03597.pdf
      optimizer = tf.train.AdagradOptimizer(learning_rate=learn_rate).minimize(loss)
      
  return (loss,optimizer)


def main(unused_argv):
  # Load training and eval data
  print("Dummy")
    # dummy declaration -- here for debugging purposes
    features = np.float32(np.zeros((5,1024)))
    # same has to be be done for trg_map
    trg_map = tf.reshape(np.float32(np.zeros((5,1024))), [-1, img_size[0],img_size[1], 1])
  
  
  # Initialize dictionary to store layers weight & bias
  
  # initialize variables
  
  # get epochs and training parameters
  
  # running the training cycle
  

if __name__ == "__main__":
  tf.app.run()
