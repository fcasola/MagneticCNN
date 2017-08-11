"""
Module creating the training data for a convolutional neural network
learning about propagation of magnetic fields in space

written by F. Casola, Harvard University - fr.casola@gmail.com
"""
import os
import numpy as np
import math as mt
import scipy.signal as sc
import argparse as ap
import h5py
import progressbar 


def Array_pad(xax,yax,mx,my,mz,dnv):
    """
    Padding the x,y-axes and the magnetization maps 
    Done to avoid finite-size effects
    """
    ##
    # x and y axes
    ##
    Ng = 30
    Extension_gr = Ng*dnv
    ## center axes
    xax_sh = xax-np.mean(xax)
    yax_sh = yax-np.mean(yax)
    ## axes span and last step
    x_ext = xax[-1]-xax[0]
    y_ext = yax[-1]-yax[0]
    dx = xax[-1]-xax[-2]
    dy = yax[-1]-yax[-2]
    ## compute extension
    ext_x = (Extension_gr-x_ext)*((Extension_gr-x_ext)>0)
    ext_y = (Extension_gr-y_ext)*((Extension_gr-y_ext)>0)
    Nptsx = int(ext_x/dx)
    Nptsy = int(ext_y/dy)    
    ## padding X    
    str_x = xax[0]-Nptsx*dx
    end_x = xax[-1]+Nptsx*dx
    x_long = np.pad(xax_sh,Nptsx,'linear_ramp',end_values=(str_x,end_x))
    ## padding Y
    str_y = yax[0]-Nptsy*dy
    end_y = yax[-1]+Nptsy*dy
    y_long = np.pad(yax_sh,Nptsy,'linear_ramp',end_values=(str_y,end_y))
    
    ##
    # mx,my,mz - zero padding out of boundaries for convergence
    ##
    mx_long = np.pad(mx,((Nptsy,Nptsy),(Nptsx,Nptsx)),'constant')
    my_long = np.pad(my,((Nptsy,Nptsy),(Nptsx,Nptsx)),'constant')
    mz_long = np.pad(mz,((Nptsy,Nptsy),(Nptsx,Nptsx)),'constant')
    
    return (x_long,y_long,mx_long,my_long,mz_long,Nptsx,Nptsy)
  

def Compute_Bz(xax,yax,t,Ms,dnv,mx,my,mz):
    """
    This function computes the stray field Bz
    using the real-space formalism introduced in the paper:
        https://arxiv.org/abs/1611.00673
    
    Input parameters are:
        *xax - x-axis, 1xN numpy vector in m
        *yax - y-axis, 1xN numpy vector in m
        *t - magnetic layer thickenss in m
        *Ms - Saturation Magnetization in A/m
        *dnv - Distance of the sensor in m
        *mx - X-component of Magnetization (adimensional)
        *my - Y-component of Magnetization (adimensional)
        *mz - Z-component of Magnetization (adimensional)       
        
    Returns
    ----------
        - Bz, the out of plane component of the magnetic field
    """    
    # Extending the domain
    x_long,y_long,mx_long,my_long,mz_long,Nptsx,Nptsy = Array_pad(xax,yax,mx,my,mz,dnv)
    # Computing PSFs (Point Spread Functions)
    # Evaluating shape of the resolution functions
    # for the convolution    
    rx,ry = np.meshgrid(x_long,y_long)
    rtot = np.sqrt(rx**2+ry**2)
    alphaxy = (1/2/mt.pi)*(dnv*t)/((dnv**2 + rtot**2)**(3/2))
    alphaz = (1/2/mt.pi)*t/((dnv**2 + rtot**2)**(1/2))
    
    # Evaluating divergence and Laplacian
    # We account for possible non-equidistant grid steps
    
    dx_l = np.gradient(x_long)
    dy_l = np.gradient(y_long)
    # divergence
    dmxx,_ = np.gradient(mx_long)
    _,dmyy = np.gradient(my_long)
    Divexy = dmxx/dx_l + dmyy/dy_l
    # laplacian
    dmzy,dmzx = np.gradient(mz_long)
    _,dmz_xx =  np.gradient(dmzx/dx_l)
    dmz_yy,_ =  np.gradient(dmzy/dy_l)
    laplacZ = dmz_xx/dx_l + dmz_yy/dy_l
    
    # Convolutions: convolving mxy and mz derivatives    
    d2x,d2y = np.meshgrid(dx_l,dy_l)
    magn_chrg = sc.convolve2d(alphaxy,Divexy,'same','wrap')*d2x*d2y
    lapl_mz = sc.convolve2d(alphaz,laplacZ,'same','wrap')*d2x*d2y
    
    # unpadding: remove extradimension after convolution through slicing
    magn_chrg_sl = magn_chrg[Nptsy:-Nptsy,Nptsx:-Nptsx]
    lapl_mz_sl = lapl_mz[Nptsy:-Nptsy,Nptsx:-Nptsx]
    
    # rescale the output and get meaningful units
    mu0 = 4*mt.pi*1e-7
    Bz = -0.5*(mu0*Ms)*(lapl_mz_sl + magn_chrg_sl)
    
    return Bz

def save_to_hdf5(dic, filename):
    """
    Compact way to save to hierarchical data format, Part I
    
    Input parameters are
        *dic: Dictionary, in the form {key: numerical or string item}
        *filename: full *.h5 name of the file to save 
        
    """
    with h5py.File(filename, 'w') as h5file:
        save_dict_contents(h5file, '/', dic)

def save_dict_contents(h5file, path, dic):
    """
    Compact way to save to hierarchical data format, Part II
    """
    for key, item in dic.items():
        if isinstance(item, (np.ndarray, np.int64, np.float64, str, bytes)):
            h5file[path + key] = item
        else:
            raise ValueError('Cannot save %s type'%type(item))

    
if __name__ == "__main__":
    # Create training data for the magnetic convolutional neural net (MCNN)
    # Parsing the input
    parser = ap.ArgumentParser(description='Create training data for the MCNN.')
    parser.add_argument('-o',"--Output", metavar='',help='Filename for the hdf5 output', \
                        required=True,type=str)
    parser.add_argument('-n', '--Ndata', metavar='', help='Number of training samples (integer)', \
                        default=1000,type=int)
    args = vars(parser.parse_args())
    
    # Processing
    # Test parameters for training computation
    # Note: because scale invariance for the CNN is imposed via scaling below,
    # these parameters are arbitrary
    Dtr=(32,32) # Fixed dimensions for the Training data
    dnv = 10e-9 # distance of the sensor
    t = 1e-9 # thickness of the magnetic layer
    Ms = 1 # Saturation magnetization, placeholder constant
    Ndense = 4*Dtr[0] # Denser grid to avoid numerical errors
    x_var = np.linspace(-(Dtr[0]//2)*dnv,(Dtr[0]//2)*dnv,Ndense)# spatial dimension of the map

    # create a generator that runs over Ndata, progressively saving into an hdf5 file
    # the generator selects arbitrary magnetization configurations
    Ntrain = args["Ndata"]
    # Initialize the dictionary
    print('Dictionary Initialization')
    DictInit = {'Theta': np.zeros([1,Ntrain]), 'Phi': np.zeros([1,Ntrain]), \
                'Bz': np.zeros([Ntrain,Dtr[0]*Dtr[1]]), 'mloc': np.zeros([Ntrain,Dtr[0]*Dtr[1]])}
    # Compute training dataset
    print('Compute training dataset')
    progress = progressbar.ProgressBar()
    for i in progress(range(Ntrain)):
        # sample directions uniformly 
        # http://corysimon.github.io/articles/uniformdistn-on-sphere/
        
        # sample phase space
    
    # create a folder if not existing called data/Training/
    directory = 'data\\Training'
    root_dir = os.getcwd() 
    if not os.path.exists(directory):    
        os.makedirs(directory)    
         
    data_filename = os.path.join(directory,args["Output"] + '.h5')
    
    # Save the dictionary into hdf5 format        
    #dict = {'field 1': np.array([1,2,3]), 'field 2': np.array([[3,4,5],[3,4,5],[3,4,5]])}
    print('Saving data as %s'%data_filename)
    save_to_hdf5(dict, data_filename)
            
        
    