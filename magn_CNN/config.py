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
k_filt_str_lyr3 = json.loads(config.get("model","k_filt_str_lyr3"))
Depth_lyr3 = config.getint("model", "Depth_lyr3")

#training
learn_rate = config.getfloat("training", "learn_rate")
epochs = config.getint("training", "epochs")
batch_size = config.getint("training", "batch_size")
Train_2_Valid = config.getfloat("training", "Train_2_Valid")
Multiplier = config.getfloat("training", "Multiplier")

#cluster mode
Script_mode = config.getboolean("cluster", "Script_mode")
Dst_ovwr = config.getboolean("cluster", "Dst_ovwr")
Model_ovwr = config.getboolean("cluster", "Model_ovwr")

#paths
write_train_path = config.get("paths", "save_train_set")
save_learned = config.get("paths", "save_learned")
save_predictions = config.get("paths", "save_predictions")

# create a dictionary that stores them all
Config_dic= {"Ntrain_ex": Ntrain_ex,
		"img_size": img_size,
		"finer_grid": finer_grid,
		"rmax_v": rmax_v,
		"act_type": act_type,
		"k_filt_str_lyr1": k_filt_str_lyr1,
		"Depth_lyr1": Depth_lyr1,
		"Pool1_str_lyr1": Pool1_str_lyr1,
		"Pool2_str_lyr1": Pool2_str_lyr1,
		"k_filt_str_lyr2": k_filt_str_lyr2,
		"Depth_lyr2": Depth_lyr2,
		"k_filt_str_lyr3": k_filt_str_lyr3,
		"Depth_lyr3": Depth_lyr3,
		"learn_rate": learn_rate,
		"epochs": epochs,
		"batch_size": batch_size,
		"Train_2_Valid": Train_2_Valid,
		"Multiplier": Multiplier,
		"Script_mode": Script_mode,
		"Dst_ovwr": Dst_ovwr,
		"Model_ovwr": Model_ovwr,		
		"write_train_path": write_train_path,
		"save_learned": save_learned,
		"save_predictions": save_predictions}

