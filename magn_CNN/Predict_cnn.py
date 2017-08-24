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
import time
# personal modules
from config import *
from config_example import *
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
    
def check_write_file(map2write,fname,limw):
    """
    write predicted map back to png
    """
    fname_full = Config_dic["save_predictions"] + '\\' + fname + '.png'
    # Check directory 
    if not os.path.exists(Config_dic["save_predictions"]):        
        os.makedirs(Config_dic["save_predictions"])        
    # Save the figure    
    min_matr = limw[0]
    max_matr = limw[1]
    # convert the map
    if np.any(map2write>max_matr) or np.any(map2write<min_matr):
        print("\nWarning! predicted m nominally out of limits. Clipping imposed.")            
        map2write[map2write>max_matr]=max_matr
        map2write[map2write<min_matr]=min_matr       
    mapPngready = (0 + 255*(map2write-min_matr)/(max_matr-min_matr)).astype(int)
    # Save as png
    scp.imsave(fname_full, mapPngready)
    

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
    directory = Config_dic["save_learned"]+'/'  
    list_of_files = []
    if os.path.exists(directory): 
        for fname in os.listdir(directory):
            if fname.endswith(extension_f):  
                list_of_files.append(os.path.join(directory,fname).replace("\\","/"))
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
        print('1/2 - Loading trained tensorflow variables.\n')
        # Recreate the network
        saver = tf.train.import_meta_graph(filelearning)      
        # load the parameters for the trained graph
        directory = Config_dic["save_learned"]+'\\' 
        saver.restore(sess, tf.train.latest_checkpoint(directory))
        # Get output function from the saved collection		
        output_fn = tf.get_collection("output_net")[0]    
        # Compute the actual prediction
        print('\n2/2 - Computing prediction.\n')
        outbasedonin = sess.run(output_fn, feed_dict = {'x:0': input_graph})
        # Reshape according to original pixel size
        predmap = outbasedonin.reshape(Config_dic["img_size"])    
        # reinterpolate in 2d
        f_int_2 = scpi.interp2d(x0,y0,predmap,kind='linear',bounds_error=False)
        m_interp_in = f_int_2(x_var,y_var)     
    # return m interpolated
    return m_interp_in  
    

def initialize_example():
    '''
    Example initialization 
    '''
    # Welcome message
    print('-'*10 + 'Running test-mode' + '-'*10 + '\n')    
    print('Generating artificial input/output of the cnn based on config_example.py.\n\
    After that, the code runs a prediction based on the trained cnn.\n')
    # define dictionary of parameters
    pars_ex = {}
    # polar angle of the magnetization
    pars_ex['Polar_angle'] = Config_dic_example['Polar_angle']*(mt.pi/180)
    # azimuthal angle of the magnetization
    pars_ex['Azimuthal_angle'] = Config_dic_example['Azimuthal_angle']*(mt.pi/180)
    # distance of the sensor
    pars_ex['dist_pl'] = Config_dic_example['dist_pl']
    # thickness of the magnetic layer (must be <dnv)
    pars_ex['thickness_layer'] = Config_dic_example['thickness_layer']
    # Saturation magnetization in A/m
    pars_ex['Saturat_magnetiz'] = Config_dic_example['Saturat_magnetiz']
    # domain axes
    pars_ex['xy_range'] = Config_dic_example['xy_range']
    # ellipse properties
    e_max = Config_dic_example['e_max']*pars_ex['dist_pl'] # ellipse semiaxis dimensionality
    e_ax = [1,Config_dic_example['e_min']/Config_dic_example['e_max']] # major and minor ellipse semiaxes in e_amax units
    e_tilt = [Config_dic_example['e_tilt']*(mt.pi/180)] # tilt of the ellipse
    e_centr = Config_dic_example['e_center'] # ellipse center (scaled to x,y-span)
    
    # Define the shape and print it to a file    
    numpxl = pars_ex['xy_range']
    dnv = pars_ex['dist_pl']
    t = pars_ex['thickness_layer']
    Ms = pars_ex['Saturat_magnetiz'] 
    denser_pt = Config_dic['finer_grid'] 
    theta = pars_ex['Polar_angle']
    phi = pars_ex['Azimuthal_angle']
    
    # spatial dimension of the map
    x_var = np.linspace(-(numpxl[0]//2)*dnv,(numpxl[0]//2)*dnv,numpxl[0]*denser_pt)
    y_var = np.linspace(-(numpxl[1]//2)*dnv,(numpxl[1]//2)*dnv,numpxl[1]*denser_pt)
    xx,yy = np.meshgrid(x_var,y_var)
    print('1/2 - Creating the magnetization shape and computing the field\n')
    # create random shapes to train the CNN
    mabs = ct.return_shape(e_max,e_ax+e_tilt+e_centr,[numpxl[0]*denser_pt, numpxl[1]*denser_pt],xx,yy)
    # compute the stray field from this map
    mx = mabs*mt.sin(theta)*mt.cos(phi)
    my = mabs*mt.sin(theta)*mt.sin(phi)
    mz = mabs*mt.cos(theta)             
    bz_test = ct.Compute_Bz(x_var-np.mean(x_var),y_var-np.mean(y_var),t,Ms,dnv,mx,my,mz)
    
    # Store axes and Bz limits
    # Tesla to Gauss conversion
    Tes2Gauss = 1e4
    bz_test *= Tes2Gauss
    # Store axes anf field limits of the maps
    pars_ex['bz_ext'] = [bz_test.min(), bz_test.max()]
    pars_ex['axes_ext'] = [(x_var[-1] - x_var[0]), (y_var[-1] - y_var[0])]
    
    # print both magnetization and Bz to a test file
    print('2/2 - Saving images in ' + Config_dic["save_predictions"])
    pars_ex['file_bz'] = 'Test_bz_'+time.strftime("%d_%m_%Y")
    
    check_write_file(bz_test,pars_ex['file_bz'],pars_ex['bz_ext'])
    check_write_file(mabs,'Test_m_'+time.strftime("%d_%m_%Y"),[0,1])
    
    print('\nDone! Look in folder ' + Config_dic["save_predictions"]) 
    print('-'*10 + '-'*16 + '-'*10 + '\n')        
    
    return pars_ex
    

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
    parser.add_argument('-e',"--example",default=False, action="store_true", \
                        help="Creates exemplary Bz data/label for the cnn using a test shape specified in the ''Config'' folder.", required=False)
    parser.add_argument('-i', "--Image", metavar='', help="Image name for the stray field.\nThe Image must be in the ''Predictions'' folder.", required=False)
    parser.add_argument('-r','--XYrange', nargs=2, metavar=('',''),
                   help='Spcify width and height of the stray field image [in meters].',required=False,type=float)   
    parser.add_argument('-c','--Caxrange', nargs=2, metavar=('',''),
                   help='Spcify minimum and maximum value of the field [in Gauss].',required=False,type=float)   
    parser.add_argument('-d', '--Distance', metavar='', help='Distance of the stray field to the magnetization plane [in meters].', \
                         required=False,type=float)
    parser.add_argument('-t', '--thickness', metavar='', help='Thickness of the magnetic layer [in meters].', \
                         required=False,type=float)
    parser.add_argument('-ms', '--Ms', metavar='', help='Maximum scalar value for the nominal saturation magnetization of\nthe layer [in A/m].', \
                         required=False,type=float)    
    args = vars(parser.parse_args())    

    if args['example'] is True:
        dic_pars_exe= initialize_example()
        args['Image'] = dic_pars_exe['file_bz']
        args['XYrange'] = dic_pars_exe['axes_ext'] 
        args['Caxrange'] = dic_pars_exe['bz_ext']
        args['Distance'] = dic_pars_exe['dist_pl']
        args['thickness'] = dic_pars_exe['thickness_layer']
        args['Ms'] = dic_pars_exe['Saturat_magnetiz']
        
      
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
            check_write_file(pred_cnn,args['Image'] + '_m_predicted',[0,1])
            
    #end
    print('Done! Look in folder ' + Config_dic["save_predictions"]) 
    print('-'*10 + '-'*16 + '-'*10 + '\n')        
                
    

    
    
    
    
    