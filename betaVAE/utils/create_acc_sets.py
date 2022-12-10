# -*- coding: utf-8 -*-
# /usr/bin/env python3
#
"""
The aim of this script is to generate csv files of subjects for each set of
data of ACC.
generation of the lists of associated subjects.
"""

import os
import pandas as pd


def create_acc_sets():
    """
    Reads different embeddings of contrastive part and generates equivalent
    subjects lists. 
    """
    root_dir = '/path/to/embeddings'

    save_dir = 'path/to/save/folder'

    df_full_sub = pd.read_csv(os.path.join(root_dir, 'full_embeddings.csv'))
    full_sub = df_full_sub['ID']
    full_sub.to_csv(os.path.join(save_dir, 'full_subjects.csv'))

    df_test_sub = pd.read_csv(os.path.join(root_dir, 'test_embeddings.csv'))
    test_sub = df_test_sub['ID']
    test_sub.to_csv(os.path.join(save_dir, 'test_subjects.csv'))

    df_train_sub = pd.read_csv(os.path.join(root_dir, 'train_embeddings.csv'))
    train_sub = df_train_sub['ID']
    train_sub.to_csv(os.path.join(save_dir, 'train_subjects.csv'))

    df_val_sub = pd.read_csv(os.path.join(root_dir, 'val_embeddings.csv'))
    val_sub = df_val_sub['ID']
    val_sub.to_csv(os.path.join(save_dir, 'val_subjects.csv'))



if __name__ == '__main__':
    create_acc_sets()
