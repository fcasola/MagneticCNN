"""
Convolutional Neural Network model for magnetism problems

written by F. Casola, Harvard University - fr.casola@gmail.com
"""

import numpy as np
import tensorflow as tf
import math as mt
import progressbar 
from tensorflow.contrib import learn
# personal modules
from config import *
# testing 
import Create_TrainingCNN as ct
import matplotlib.pyplot as plt

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

def cnn_model_fn(features, trg_map, mode, Config_dic):   
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
        *Config_dic - Properties of the model; read from config file
        
    Returns
    ----------
        - the tuple (loss,optimizer,init), representing the type of loss function 
        and the optimizer method used. init is an Op that initializes global variables
  """
  # Reshape tensors to 4-D tensor: [batch_size, width, height, channels]
  # Magnetic field images are img_size[0]ximg_size[1] pixels, and have one channel
  
  #input_layer = tf.reshape(features, [-1, Config_dic["img_size"][0],Config_dic["img_size"][1], 1])
  #trg_map_rsh = tf.reshape(trg_map, [-1, Config_dic["img_size"][0],Config_dic["img_size"][1], 1])
  
  x = tf.placeholder(tf.float32, [Config_dic["batch_size"], features.shape[1]])
  y = tf.placeholder(tf.float32, [Config_dic["batch_size"], trg_map.shape[1]])  
  input_layer = tf.reshape(x, [-1, Config_dic["img_size"][0],Config_dic["img_size"][1], 1])
  trg_map_rsh = tf.reshape(y, [-1, Config_dic["img_size"][0],Config_dic["img_size"][1], 1])
  
  # Selecting activation type.    
  if Config_dic["act_type"]=='relu':
     act_type = tf.nn.relu
  else:
     act_type = tf.nn.sigmoid

  # LAYER 1 ---------------         
  # Convolutional layer # 1    
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=Config_dic["Depth_lyr1"],
      kernel_size=Config_dic["k_filt_str_lyr1"][0:2],
      padding="same",
      activation=act_type)  
  # Pooling1, layer 1 - note - There are no trainable parameters in a max-pooling layer
  pool1_lyr1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[Config_dic["Pool1_str_lyr1"][0]]*2, 
                                       strides=Config_dic["Pool1_str_lyr1"][1])
  # Pooling2, layer 1 - note - There are no trainable parameters in a max-pooling layer
  pool2_lyr1 = tf.layers.max_pooling2d(inputs=pool1_lyr1, pool_size=[Config_dic["Pool2_str_lyr1"][0]]*2, 
                                  strides=Config_dic["Pool2_str_lyr1"][1])
  
  # LAYER 2 ---------------         
  conv2 = tf.layers.conv2d(
      inputs=conv1,
      filters=Config_dic["Depth_lyr2"],
      kernel_size=Config_dic["k_filt_str_lyr2"][0:2],
      padding="same",
      activation=act_type)  
  # Pooling1, layer 2
  pool1_lyr2 = tf.layers.conv2d(
      inputs=pool1_lyr1,
      filters=Config_dic["Depth_lyr2"],
      kernel_size=Config_dic["k_filt_str_lyr2"][0:2],
      padding="same",
      activation=act_type)  
  # Pooling2, layer 2
  pool2_lyr2 = tf.layers.conv2d(
      inputs=pool2_lyr1,
      filters=Config_dic["Depth_lyr2"],
      kernel_size=Config_dic["k_filt_str_lyr2"][0:2],
      padding="same",
      activation=act_type)  

  # LAYER 3 ---------------         
  conv3 = tf.layers.conv2d(
      inputs=conv2,
      filters=Config_dic["Depth_lyr3"],
      kernel_size=Config_dic["k_filt_str_lyr3"][0:2],
      padding="same",
      activation=act_type)  
  # Pooling1, layer 2
  pool1_lyr3 = tf.layers.conv2d(
      inputs=pool1_lyr2,
      filters=Config_dic["Depth_lyr3"],
      kernel_size=Config_dic["k_filt_str_lyr3"][0:2],
      padding="same",
      activation=act_type)  
  # Pooling2, layer 2
  pool2_lyr3 = tf.layers.conv2d(
      inputs=pool2_lyr2,
      filters=Config_dic["Depth_lyr3"],
      kernel_size=Config_dic["k_filt_str_lyr3"][0:2],
      padding="same",
      activation=act_type)  

  
  # Do upscaling and combining
  # tensors conv2,pool1_lyr3 and pool2_lyr3 
  # Pooling MUST rescale the map by powers of 2 by design
  
  # Upscaling
  factor1 = mt.log(Config_dic["Pool1_str_lyr1"][0],2)
  factor2 = mt.log(Config_dic["Pool2_str_lyr1"][0]*Config_dic["Pool2_str_lyr1"][0],2)
  if ((factor1%1)!=0 or (factor2%1)!=0)==True:
      raise ValueError('Pooling size/sizes not multiple of 2')        
  ndim1 = [int(Config_dic["img_size"][0]/(2**factor1)), \
           int(Config_dic["img_size"][1]/(2**factor1)),Config_dic["Depth_lyr3"]] 
  ndim2 = [int(Config_dic["img_size"][0]/(2**factor2)), \
           int(Config_dic["img_size"][1]/(2**factor2)),Config_dic["Depth_lyr3"]]
  US_pool1_lyr3 =  upscaling(pool1_lyr3,factor1,ndim1)
  US_pool2_lyr3 =  upscaling(pool2_lyr3,factor2,ndim2)

  # Combining Upscaled pooling layers with the regular Conv layer
  Combined_output = conv3 + US_pool1_lyr3 + US_pool2_lyr3
  
  # LAYER 4 ---------------         
  # Apply a 1x1 convolution to act with ReLu, as in https://arxiv.org/pdf/1607.03597.pdf
  conv4 = tf.layers.conv2d(
      inputs=Combined_output,
      filters=Config_dic["Depth_lyr3"],
      kernel_size=Config_dic["k_filt_str_lyr3"][0:2],
      padding="same",
      activation=act_type)   
  
  # LAYER 5 ---------------         
  # Apply a 1x1 convolution to reduce feature size
  conv5 = tf.layers.conv2d(
      inputs=conv4,
      filters=1,
      kernel_size=Config_dic["k_filt_str_lyr3"][0:2],
      padding="same",
      activation=act_type) 
  
  # The loss function
  loss = None
  optimizer = None
  
  if mode != learn.ModeKeys.INFER:  
      loss = tf.reduce_mean(tf.square(trg_map_rsh - conv5))
      
  if mode == learn.ModeKeys.TRAIN:
      # Adam optimizer as in https://arxiv.org/pdf/1607.03597.pdf
      optimizer = tf.train.AdagradOptimizer(learning_rate=Config_dic["learn_rate"]).minimize(loss)
      
  # initialize all variables
  init = tf.global_variables_initializer()   

  return (loss,optimizer,init,x,y)

# write a function that saves to file using saver = tf.train.Saver()
# write a function that loads from file using saver.restore(sess, "/tmp/model.ckpt")



def main(unused_argv):
  # Load training and eval data
  print("training")
  filename = "..\data\Training\Training_set_12_08_2017.h5"
  loaddataset =  ct.load_from_hdf5(filename)
  itemsel=200  
  #bz_sel = loaddataset["Bz"][0:itemsel][:].reshape((-1,mapdim,mapdim))
  #m_sel = loaddataset["mloc"][0:itemsel][:].reshape((-1,mapdim,mapdim))
  features = np.float32(loaddataset["Bz"][0:itemsel][:])
  trg_map = np.float32(loaddataset["mloc"][0:itemsel][:])
  
  features = np.float32(loaddataset["Bz"][0:Config_dic["batch_size"]][:])
  trg_map = np.float32(loaddataset["mloc"][0:Config_dic["batch_size"]][:])
  
  
  #x = tf.placeholder(tf.float32, [Config_dic["batch_size"], features.shape[1]])
  #y = tf.placeholder(tf.float32, [Config_dic["batch_size"], trg_map.shape[1]])
  
  #phi = loaddataset["Phi"][itemsel][:]
  #tht = loaddataset["Theta"][itemsel][:]

  # Initialize the model and the related variables
  tf.reset_default_graph()  
  loss,optimizer,init,x,y = cnn_model_fn(features, trg_map, 'train',Config_dic)
  # Getting the list of all trainable variables
  tvar = tf.trainable_variables()  
  # starting the session
  with tf.Session() as sess: 
      sess.run(init)
      # training cycle; have to add
      print('Training cycle started...')
      progress = progressbar.ProgressBar()
      for epoch in progress(range(Config_dic["epochs"])):
          avg_cost = 0
          total_batch = int(features.shape[0]/Config_dic["batch_size"])
          for i in range(total_batch):
              batch_feat = features[i*Config_dic["batch_size"]: \
                                    (i+1)*Config_dic["batch_size"]]
              batch_trg = trg_map[i*Config_dic["batch_size"]: \
                                  (i+1)*Config_dic["batch_size"]]
              # Minimizer   
              _, BatchLoss = sess.run([optimizer, loss], feed_dict={x: batch_feat, y: batch_trg})
    
              avg_cost += BatchLoss/total_batch
              
          print("Epoch:", (epoch+1), "cost =", "{:.5f}".format(avg_cost))              
  
      print("Training complete!")

if __name__ == "__main__":
  tf.app.run()
