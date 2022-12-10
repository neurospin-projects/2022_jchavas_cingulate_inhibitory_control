#!/usr/bin/env python
# -*- coding: utf-8 -*-
#import logging
import os
import glob
import yaml

import omegaconf
from omegaconf import DictConfig
from omegaconf import OmegaConf
log = logging.getLogger(__name__)

import pandas as pd


def process_config(config) -> DictConfig:
    """Does whatever operations on the config file
    """

    log.info(OmegaConf.to_yaml(config))
    log.info("Working directory : {}".format(os.getcwd()))
    config.input_size = eval(config.input_size)
    log.info("config type: {}".format(type(config)))
    if "pretrained_model_path" not in config:
        config.pretrained_model_path = None
    return config


def create_accessible_config(keys_to_keep, config_path):
    """Create a yaml file with only targeted keys at the root of the training folder.
    Inputs:
        - keys_to_keep: config parameters you want to have access to easily.
        - config_path: path where is stored the full config file."""
    with open(config_path, 'r') as file:
        config_dict = yaml.load(file, Loader=yaml.FullLoader)

    partial_config = {}

    for key in config_dict.keys():
        if key in keys_to_keep:
            partial_config[key] = config_dict[key]
    
    save_path = '/' + os.path.join(*config_path.split("/")[:-2])
    print(save_path)
    with open(save_path+'/partial_config.yaml', 'w') as file:
        yaml.dump(partial_config, file)


def get_config_diff(dir_path, whole_config=False, save=True, verbose=False):
    """Get the parameters in config (or only in the partial config) that changed between the models
    of the targeted folder.
    /!\ Probably poorly optimised.
    Inputs:
        - dir_path: path to directory where models to compare are stored
        - whole_config: bool telling if all the config parameters have to be compared. If not, only 
        the ones in the partial_config are compared."""

    # number of sub directories (excluding files)
    only_dirs = [name for name in os.listdir(dir_path) 
                 if (not (os.path.isfile(dir_path+'/'+name))  # condition to be a folder
                 and (glob.glob(dir_path+'/'+name + r'/*config.yaml') != []))] # condition to be a model
    n_subdir = len(only_dirs)
    if verbose:
        print(f'{n_subdir} subdirs:', only_dirs)

    if n_subdir == 1:
        # return and not error not to stop the train function
        return("The chosen directory contains only one training: can't compare the parameters.")

    global_df = pd.DataFrame()

    # loop over subdirectories
    for subdir in only_dirs:
        # choose which config files to compare
        if whole_config:
            config_file = dir_path+'/'+subdir + '/.hydra/config.yaml'
        else:
            config_file = glob.glob(dir_path+'/'+subdir + r'/*config.yaml')
            if len(config_file) == 0:
                raise ValueError(f'No summarized config file in {subdir} folder.')
            elif len(config_file) > 1:
                raise ValueError(f'Several possible config in {subdir} folder. Change names such \
    as only the file you want ends by "config.yaml".')
            else:
                config_file = config_file[0]
        
        # load and convert dict values into strings
        with open(config_file, 'r') as file:
            config_dict = yaml.load(file,  Loader=yaml.FullLoader)
        for key in config_dict:
            config_dict[key] = str(config_dict[key])
        
        # add config to a global dataframe
        config_df = pd.DataFrame.from_dict({subdir: config_dict}, orient='index')
        global_df = pd.concat([global_df, config_df], axis=0)


    # get columns that have variations
    diff_keys = []

    for key in global_df.columns:
        if verbose:
            print(key)
        if (global_df[key] != global_df[key][0]).any():
            diff_keys.append(key)
    
    # keep only these columns and sort the indexes
    global_df = global_df[diff_keys].sort_index()

    if save:
        global_df.to_csv(dir_path+'/config_diff.csv')
    else:
        return diff_keys, global_df
