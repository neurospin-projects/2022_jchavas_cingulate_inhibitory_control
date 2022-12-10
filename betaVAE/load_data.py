# -*- coding: utf-8 -*-
# /usr/bin/env python3
#
"""
Tools in order to create pytorch dataloaders
"""
import os
import sys
import re

import pandas as pd
import numpy as np
from preprocess import *


def create_subset(config):
    """
    Creates dataset HCP_1 from HCP data
    Args:
        config: instance of class Config
    Returns:
        subset: Dataset corresponding to HCP_1
    """
    train_list = pd.read_csv(config.subject_dir, header=None, usecols=[0],
                             names=['subjects'])
    train_list['subjects'] = train_list['subjects'].astype('str')

    tmp = pd.read_pickle(os.path.join(config.data_dir, "Rskeleton.pkl")).T
    tmp.index.astype('str')
    tmp['subjects'] = [re.search('(\d{6})', tmp.index[k]).group(0) for k in range(
                        len(tmp))]
    tmp = tmp.merge(train_list, left_on = 'subjects', right_on='subjects', how='right')

    filenames = list(train_list['subjects'])

    subset = SkeletonDataset(dataframe=tmp, filenames=filenames)

    return subset


def create_test_subset(config):
    """
    Creates test dataset from ACC database
    Args:
        config: instance of class Config
    Returns:
        subset: Dataset corresponding to ACC dataset
    """

    tmp = pd.read_pickle(os.path.join(config.acc_subjects_dir, "Rskeleton.pkl")).T
    tmp.index.astype('str')

    re_expr = '/crops/2mm/CINGULATE/mask/Rcrops/(.*)'

    tmp['subjects'] = [re.search(re_expr, tmp.index[k]).group(1) for k in range(
                        len(tmp))]

    filenames = list(tmp['subjects'])

    subset_test = SkeletonDataset(dataframe=tmp, filenames=filenames)

    return subset_test
