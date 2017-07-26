# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 17:35:37 2017
@author: Francesco

"""

import argparse as ap

if __name__ == "__main__":
    '''
    This code implements convolutional Neural Networks 
    to obtain the magnetic structure producing a given 
    stray magnetic field.

    '''    
    
    # Parse the command line arguments
    parser = ap.ArgumentParser()
    parser.add_argument('-t',"--test", help="Running in Test mode", required=False)
    args = vars(parser.parse_args())
    print("Program is terminated")
    
