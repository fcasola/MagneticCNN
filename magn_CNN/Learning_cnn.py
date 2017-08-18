"""
Convolutional Neural Network model for magnetism problems

written by F. Casola, Harvard University - fr.casola@gmail.com
"""

import numpy as np
import tensorflow as tf
import math as mt
import progressbar 
import os
import time
from tensorflow.contrib import learn
import Create_TrainingCNN as ct
# personal modules
from config import *

# Reduce verbosity of tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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

def cnn_model_fn(mode, Config_dic):   
  """
    Model function for magnetic CNN.
    This function implements the convolutional neural net model 
    discussed in arxiv.org/pdf/1607.03597.pdf.
    Input parameters are:
        *mode - specifies the running mode. Definitions are
                according to the standard learn.ModeKeys keys.
                ** 'train': training mode.
                ** 'eval': evaluation mode.
                ** 'infer': inference mode.
        *Config_dic - Properties of the model; read from config file
        
    Returns
    ----------
        - the tuple (loss,optimizer,init,x,y), representing the type of loss function 
        and the optimizer method used. 
        init is an Op that initializes global variables.
        x,y represent input and output of the cnn
  """
  # Number of pixels in the image
  Numpixels = Config_dic["img_size"][0]*Config_dic["img_size"][1]
  # Tensorflow placeholders for the input and output of the cnn
  x = tf.placeholder(tf.float32, [None, Numpixels], name='x')
  y = tf.placeholder(tf.float32, [None, Numpixels], name='y')  
  # Reshaping the input
  input_layer = tf.reshape(x, [-1, Config_dic["img_size"][0],Config_dic["img_size"][1], 1])
  
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
  init = 0
  
  # Prepare for loss calculation
  out_rshp = tf.reshape(conv5, [-1, Numpixels])
  
  
  if mode != learn.ModeKeys.INFER:  
      # reduce_mean does not depend on batch size,
      # contrary to reduce_sum
      loss = tf.reduce_mean(tf.square(y - out_rshp))
      
  if mode == learn.ModeKeys.TRAIN:
      # Adam optimizer as in https://arxiv.org/pdf/1607.03597.pdf
      optimizer = tf.train.AdagradOptimizer(learning_rate=Config_dic["learn_rate"]).minimize(loss)
      # Adding outputs to collection
      tf.add_to_collection("output_net", out_rshp)
      tf.add_to_collection("loss_net", loss)
     
  # initialize all variables
  init = tf.global_variables_initializer()   
    
  return (loss,optimizer,init,x,y)


def run_training(loss,optimizer,init,x,y,features,targets,session_datafile):
  # Saving the loss  
  costv=[]
  # randomly picking the data
  # seed = 128
  # rng = np.random.RandomState(seed)  
  # Starting the saver
  saver = tf.train.Saver()
  # starting the session
  with tf.Session() as sess: 
      sess.run(init)
      # training cycle;
      print('\nTraining cycle started...')
      progress = progressbar.ProgressBar()
      for epoch in progress(range(Config_dic["epochs"])):
          avg_cost = 0
          total_batch = int(features.shape[0]/Config_dic["batch_size"])
          for i in range(total_batch):
              #batch_mask = rng.choice(bz_sel.shape[0], Config_dic["batch_size"])
              batch_feat = features[i*Config_dic["batch_size"]: \
                                   (i+1)*Config_dic["batch_size"]]
              batch_trg = targets[i*Config_dic["batch_size"]: \
                                   (i+1)*Config_dic["batch_size"]]
              #batch_feat = bz_sel[batch_mask,:]
              #batch_trg = m_sel[batch_mask,:]       
              # Minimizer   
              _, BatchLoss = sess.run([optimizer, loss], feed_dict={x: batch_feat, y: batch_trg})
    
              avg_cost += BatchLoss/total_batch
              
          print("\nEpoch:", (epoch+1), "cost =", "{:.5f}".format(avg_cost))              
          costv.append(avg_cost)    
      print('Saving training')    
      save_path = saver.save(sess, session_datafile) # .ckpt file   
      print('Training complete!')
      

# write a function that loads from file using saver.restore(sess, "/tmp/model.ckpt")

if __name__ == "__main__":
  # Load training and eval data 
  # this section is hard-coded... not completed
  print('-'*10 + 'Learning algorithm' + '-'*10 + '\n')
  
  # get latest modified file in the Training dataset folder
  extension_f = '.h5'
  directory = Config_dic["write_train_path"]  
  list_of_files = []
  if os.path.exists(directory): 
   for fname in os.listdir(directory):
       if fname.endswith(extension_f):     
           list_of_files.append(os.path.join(directory,fname))
   if len(list_of_files) !=0:           
       mode = 'train'           
      # Loading from the latest file saved  
       latest_file = max(list_of_files, key=os.path.getctime)
       print('The dataset\n: ' + latest_file + '\n will be used for training\n')
       loaddataset =  ct.load_from_hdf5(latest_file)
       itemsel=int(len(loaddataset['Phi'])*3/5)  # only 2/5 data picked
       #bz_sel = loaddataset["Bz"][0:itemsel][:].reshape((-1,mapdim,mapdim))
       #m_sel = loaddataset["mloc"][0:itemsel][:].reshape((-1,mapdim,mapdim))
       bz_sel = np.float32(1e15*loaddataset["Bz"][0:itemsel][:])
       m_sel = np.float32(loaddataset["mloc"][0:itemsel][:])  
       #phi = loaddataset["Phi"][itemsel][:]
       #tht = loaddataset["Theta"][itemsel][:]   
   else:
       # folder is empty
       mode = 'exit'
  else:
   mode = 'exit'

  if mode == learn.ModeKeys.TRAIN:
    print("training started...\n")
    args={} # this should come from parsing.. to be done
    args["Output"] = 'Learned_mod_'+time.strftime("%d_%m_%Y")

    # check if trained models are there already    
    extension_f = '.meta'
    directory = Config_dic["save_learned"]
    data_filename = os.path.join(directory,args["Output"])
    Proc_furth = ct.check_proceed(directory,data_filename,extension_f)    

    if Proc_furth==1:       
      # Initialize the model and the related variables
      print('\n1/2 - Creating the cnn model.')
      tf.reset_default_graph()  
      loss,optimizer,init,x,y = cnn_model_fn(mode,Config_dic)
      # Getting the list of all trainable variables
      tvar = tf.trainable_variables()  
      # Running the training
      print('Done!\n')
      print('2/2 - Executing the training\n')
      print('Epochs: %4d Batch size: %4d Learn rate: %6.4f Dataset size: %5d \n' \
            %(Config_dic["epochs"],Config_dic["batch_size"],Config_dic["learn_rate"],itemsel))
      run_training(loss,optimizer,init,x,y,bz_sel,m_sel,data_filename)
      print('Done!\n')
    else:
        print('\n No new learning data produced.\n \
              The most recent file will be used in predictions.')            
  elif mode == 'exit':  
    print('No dataset available for training.')    
  else:
    print('No action')  
    
  print('-'*10 + '-'*16 + '-'*10 + '\n')        

