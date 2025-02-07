# Utility QGraph project settings
# Author: Jacob Cybulski, ironfrown[at]gmail.com
# Date: 2024-2025

##### Initial settings

import sys
sys.path.append('.')
sys.path.append('..')
sys.path

import os
import numpy as np
import math
import json

from utils.Files import *

##### Default project settings

class Log_Settings(object):
    LOG_NAME = 'logs'
    CASE_NAME = 'unknown'
    DATA_NAME = 'unknown'
    GRAPH_PATH = f'{LOG_NAME}/graph'
    FIGURES_PATH = f'{LOG_NAME}/figures'
    TRAIN_PATH = f'{LOG_NAME}/training'
    ANALYSIS_PATH = f'{LOG_NAME}/analysis'
    FIGURES_PATH = f'{LOG_NAME}/figures'

class Case_Settings(object):
    case_name = 'unknown'
    major_vers = 1
    minor_vers = 12

class Data_Settings(object):
    data_name = 'unknown'
    vers = 0
    n_vertices = 8
    n_edges = 0
    edge_p = 0.25
    method = 'rand' # 'scale'

class Train_Settings(object):
    n_data = 8
    n_extra = 1
    n_wires = 9
    n_layers = 3
    rot = 'Rxyz'
    mode = 'hybrid'
    
    iters = 10
    epochs = 4000
    log_interv = 1
    scaler = np.pi
    shots = 1000
    seed = 2024
    thr = 0.01


##### Manipulate setting parameters

### Find the list of settings names
def settings_names(obj):
    obj_vars = [attr for attr in dir(obj) if not callable(getattr(obj, attr)) and not attr.startswith("__")]
    return obj_vars

### Find a dictionary of settings, with their names and values
def settings_dict(obj):
    obj_vars = [attr for attr in dir(obj) if not callable(getattr(obj, attr)) and not attr.startswith("__")]
    sdict = {}
    for sname in obj_vars:
        sval = getattr(obj, sname)
        sdict[sname] = sval
    return sdict

### Save settings
def settings_save(fpath, obj):
    sdict = settings_dict(obj)
    write_json_file(fpath, sdict)

def settings_load(fpath, sobj):
    sdict = read_json_file(fpath)
    for key in sdict.keys():
        setattr(sobj, key, sdict[key])

