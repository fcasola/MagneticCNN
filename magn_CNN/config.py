"""
Module reading the configuration file

written by F. Casola, Harvard University - fr.casola@gmail.com
"""

from six.moves import configparser
import json

config = configparser.RawConfigParser()
config.read('../data/Config/config_training.cfg')

#dataset
Ntrain_ex = config.getint("Dataset", "Ntrain_ex")
img_size = json.loads(config.get("Dataset","img_size"))
finer_grid = config.getint("Dataset", "finer_grid")
rmax_v = config.getint("Dataset", "rmax_v")

#model
act_type = config.get("model", "act_type")
k_filt_str_lyr1 = json.loads(config.get("model","k_filt_str_lyr1"))
Depth_lyr1 = config.getint("model", "Depth_lyr1")
Pool1_str_lyr1 = json.loads(config.get("model","Pool1_str_lyr1"))
Pool2_str_lyr1 = json.loads(config.get("model","Pool2_str_lyr1"))
k_filt_str_lyr2 = json.loads(config.get("model","k_filt_str_lyr2"))
Depth_lyr2 = config.getint("model", "Depth_lyr2")

#paths
write_train_path = config.get("paths", "save_train_set")

#normalize = config.getboolean("hog", "normalize")
#threshold = config.getfloat("nms", "threshold")