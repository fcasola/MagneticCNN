"""
Module reading the configuration file

written by F. Casola, Harvard University - fr.casola@gmail.com
"""

from six.moves import configparser
import json

config = configparser.RawConfigParser()
config.read('../data/Config/config_training.cfg')


img_size = json.loads(config.get("Dataset","img_size"))
finer_grid = config.getint("Dataset", "finer_grid")
rmax_v = config.getint("Dataset", "rmax_v")
write_train_path = config.get("paths", "save_train_set")
#normalize = config.getboolean("hog", "normalize")
#threshold = config.getfloat("nms", "threshold")