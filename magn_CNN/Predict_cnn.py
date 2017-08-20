"""
Convolutional Neural Network model for magnetism problems

Prediction module

written by F. Casola, Harvard University - fr.casola@gmail.com
"""
import os
import numpy as np
import math as mt
import scipy.misc as scp
import argparse as ap
import scipy.interpolate as scpi
import warnings
import tensorflow as tf
# personal modules
from config import *
import Create_TrainingCNN as ct

# Reduce verbosity of tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def check_read_file(argin):
    """
    check for stray field file existence
    """
    fname = argin['Image']
    climits = argin['Caxrange']
    dnv = argin['Distance'] 
    t = argin['thickness']
    Ms = argin['Ms']
    XYrange = argin['XYrange']
    fname_full = Config_dic["save_predictions"] + '\\' + fname + '.png'
    if os.path.isfile(fname_full):
        min_matr = climits[0]
        max_matr = climits[1]
        # load the image
        imreadback = scp.imread(fname_full, mode='L')
        # convert back into numerics
        reconv_img = min_matr + (max_matr - min_matr)*(imreadback/255)
        # rescale the caxis to match cnn input requirements
        bz_scaled = -Config_dic["Multiplier"]*(reconv_img/1e4)*(dnv**3)/(1e-7*t*Ms)    
        # create x and y axes
        x_var = np.linspace(0,XYrange[0],bz_scaled.shape[1])
        y_var = np.linspace(0,XYrange[1],bz_scaled.shape[0])
        # return 
        return (x_var,y_var,bz_scaled)
    else:
        print('File does not exist.')
        return (None,None,None)
    
def check_write_file(map2write,fname):
    """
    write predicted map back to png
    """
    fname_full = Config_dic["save_predictions"] + '\\' + fname + '.png'
    if os.path.exists(Config_dic["save_predictions"]):
        min_matr = 0
        max_matr = 1
        # convert the map
        if np.any(map2write>1) or np.any(map2write<0):
            print("\nWarning! predicted m nominally out of 0-1 limits. Clipping imposed.")            
            map2write[map2write>1]=1
            map2write[map2write<0]=0       
        mapPngready = (0 + 255*(map2write-min_matr)/(max_matr-min_matr)).astype(int)
        # Save as png
        scp.imsave(fname_full, mapPngready)
    else:
        print('Writing directory does not exists.')
    

def interp2d_cnn(x_in,y_in,z_in,argin):
    '''
    Interpolate within a grid with Config_dic["img_size"] number of pixels.
    Each pixel corresponds to moving by "Distance", such as to preserve 
    scale-invariance.
    '''
    # centered old axes
    x_inc = x_in - np.mean(x_in)
    y_inc = y_in - np.mean(y_in)    
    # cnn grid definition
    D0 = Config_dic["img_size"]
    dnv = argin['Distance'] 
    x0 = np.linspace(-(D0[0]//2)*dnv,(D0[0]//2)*dnv,D0[0])
    y0 = np.linspace(-(D0[1]//2)*dnv,(D0[1]//2)*dnv,D0[1])
    # center the cnn grid
    x0_c = x0 - np.mean(x0)
    y0_c = y0 - np.mean(y0)
    # raise warning if image must be cut
    D1 = [round(argin['XYrange'][0]/dnv),round(argin['XYrange'][1]/dnv)]
    sz1 = [i*dnv for i in D0]
    sz2 = [i*dnv for i in D1]
    if sz1[0]<sz2[0] or sz1[1]<sz2[1]:
        warnings.warn("\nWarning! input image size is bigger than current cnn definition.\n\
                      Input image will be cut to satisfy requirements.\n\
                      Train a larger CNN!")
    # interpolate 
    f_int = scpi.interp2d(x_inc,y_inc,z_in,bounds_error=False)
    z_inter_out = f_int(x0_c, y0_c)                  
    return (x0,y0,z_inter_out)


def check_training_avail():
    """
    check that a training has been saved
    """
    extension_f = '.meta'
    directory = Config_dic["save_learned"]+'\\'  
    list_of_files = []
    if os.path.exists(directory): 
        for fname in os.listdir(directory):
            if fname.endswith(extension_f):  
                list_of_files.append(os.path.join(directory,fname))
    # If *.meta files have been found, load the latest modified one
    if len(list_of_files) !=0:           
        # get latest modifield file
        latest_file = max(list_of_files, key=os.path.getctime)
        # return *.meta file
        return latest_file
    else:
        print('No *.meta training file available for predictions.')
        return None
    


def predict_out(bz_scl_int,filelearning,x0,y0,x_var,y_var):
    """
    Run the prediction given the training.
    Uses latest *.meta and training available.
    """
    # Number of pixels in the image
    Numpixels = Config_dic["img_size"][0]*Config_dic["img_size"][1]
    # Cast map into proper float for tensorflow
    input_graph = np.float32(bz_scl_int.reshape(-1,Numpixels))        
    # start tensorflow session
    with tf.Session() as sess: 
        # prediction started
        print('Loading trained tensorflow variables.\n')
        # Recreate the network
        saver = tf.train.import_meta_graph(filelearning)      
        # load the parameters for the trained graph
        directory = Config_dic["save_learned"]+'\\' 
        saver.restore(sess, tf.train.latest_checkpoint(directory))
        # Get output function from the saved collection		
        output_fn = tf.get_collection("output_net")[0]    
        # Compute the actual prediction
        print('\nComputing prediction.\n')
        outbasedonin = sess.run(output_fn, feed_dict = {'x:0': input_graph})
        # Reshape according to original pixel size
        predmap = outbasedonin.reshape(Config_dic["img_size"])    
        # reinterpolate in 2d
        f_int_2 = scpi.interp2d(x0,y0,predmap,kind='linear',bounds_error=False)
        m_interp_in = f_int_2(x_var,y_var)     
    # return m interpolated
    return m_interp_in        
    
    

if __name__ == "__main__":
    # Module predicting the magnetization distribution given the stray field
    # Parsing the input
    parser = ap.ArgumentParser(description='\
    Making predictions based on the trained cnn.\n\
    The module takes as input an 8-bit pixels, black and white png image\n\
    representing the stray field and creates an image with the same resolution\n\
    representing the estimate of the cnn for a spatially varying collinear magnetization.\n\
    Notes: \n\
    1 - In using this function (see parameter definitions below), \n\
    remember that the current model holds in the limit t<<d.\n\
    2 - The b/w image containing the stray field is defined such that\n\
    b/w (0-255) is the min/max of the field in Gauss.\n\
    3 - The output is saved with similar conventions as in 2), where \n\
    now b/w is 0/1 in Ms units. Look at "*_predicted.png"', formatter_class=ap.RawTextHelpFormatter)
    parser.add_argument('-i', "--Image", metavar='', help="Image name for the stray field.\nThe Image must be in the ''Predictions'' folder.", required=True)
    parser.add_argument('-r','--XYrange', nargs=2, metavar=('',''),
                   help='Spcify width and height of the stray field image [in meters].',required=True,type=float)   
    parser.add_argument('-c','--Caxrange', nargs=2, metavar=('',''),
                   help='Spcify minimum and maximum value of the field [in Gauss].',required=True,type=float)   
    parser.add_argument('-d', '--Distance', metavar='', help='Distance of the stray field to the magnetization plane [in meters].', \
                         required=True,type=float)
    parser.add_argument('-t', '--thickness', metavar='', help='Thickness of the magnetic layer [in meters].', \
                         required=True,type=float)
    parser.add_argument('-ms', '--Ms', metavar='', help='Maximum scalar value for the nominal saturation magnetization of\nthe layer [in A/m].', \
                         required=True,type=float)    
    args = vars(parser.parse_args())    

    # Welcome message
    print('-'*10 + 'Prediction algorithm' + '-'*10 + '\n')    
    # check filename is there
    x_var,y_var,bz_read = check_read_file(args)
    
    
    if bz_read is not None:
        # test image scale and convert to Config_dic["img_size"] dimensions         
        x_long,y_long,_,_,bz_long,_,_ = ct.Array_pad(x_var,y_var,None,None,bz_read, \
                                                       args['Distance'],Config_dic["img_size"])        
        # interpolate within the proper grid
        x0,y0,bz_scl_int = interp2d_cnn(x_long,y_long,bz_long,args)   
                
        # get latest available training
        latest_file = check_training_avail()
        if latest_file != None:
            # run predictions
            pred_cnn = predict_out(bz_scl_int,latest_file,x0,y0,x_var-x_var.mean(),y_var-y_var.mean())    
            # save to file
            check_write_file(pred_cnn,args['Image'] + '_predicted')
            
    #end
    print('Done! Look in folder ' + Config_dic["save_predictions"]) 
    print('-'*10 + '-'*16 + '-'*10 + '\n')        
                
    
    
    
    
    
    
    