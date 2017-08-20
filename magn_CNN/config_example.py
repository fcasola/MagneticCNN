"""
Module reading the configuration file for the test example

written by F. Casola, Harvard University - fr.casola@gmail.com
"""
from six.moves import configparser
import json

config = configparser.RawConfigParser()
config.read('../data/Config/test_shape.cfg')

# Magnetization
Polar_angle = config.getfloat("Magnetization", "Polar_angle")
Azimuthal_angle = config.getfloat("Magnetization", "Azimuthal_angle")
thickness_layer = config.getfloat("Magnetization", "thickness_layer")
Saturat_magnetiz = config.getfloat("Magnetization", "Saturat_magnetiz")
dist_pl = config.getfloat("Magnetization", "dist_pl")
xy_range = json.loads(config.get("Magnetization","xy_range"))

# Ellipse
e_max = config.getfloat("Ellipse", "e_max")
e_min = config.getfloat("Ellipse", "e_min")
e_tilt = config.getfloat("Ellipse", "e_tilt")
e_center = json.loads(config.get("Ellipse","e_center"))

# create a dictionary that stores them all
Config_dic_example= {"Polar_angle": Polar_angle,
		"Azimuthal_angle": Azimuthal_angle,
		"thickness_layer": thickness_layer,
		"Saturat_magnetiz": Saturat_magnetiz,
		"dist_pl": dist_pl,
		"xy_range": xy_range,
		"e_max": e_max,
		"e_min": e_min,
		"e_tilt": e_tilt,
		"e_center": e_center}