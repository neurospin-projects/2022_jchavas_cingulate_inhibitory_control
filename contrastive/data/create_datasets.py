#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tools to create datasets
"""

# only if foldlabel == True
try:
    from deep_folding.brainvisa.utils.save_data import compare_npy_file_aims_files
    from deep_folding.brainvisa.utils.save_data import compare_array_aims_files
except ImportError:
    print("INFO: you cannot use deep_folding in brainvisa. Probably OK.")

from contrastive.utils.logs import set_file_logger

from contrastive.data.datasets import ContrastiveDataset
from contrastive.data.datasets import ContrastiveDataset_Visualization
from contrastive.data.datasets import ContrastiveDataset_WithLabels
from contrastive.data.datasets import ContrastiveDataset_WithFoldLabels
from contrastive.data.datasets import \
    ContrastiveDataset_WithLabels_WithFoldLabels

from contrastive.data.utils import *

log = set_file_logger(__file__)


def create_sets_without_labels(config):
    """Creates train, validation and test sets

    Args:
        config (Omegaconf dict): contains configuration parameters
    Returns:
        train_dataset, val_dataset, test_datasetset, train_val_dataset (tuple)
    """

    # Loads and separates in train_val/test skeleton crops
    train_val_subjects, train_val_data, test_subjects, test_data = \
        extract_data(config.numpy_all, config)

    # Loads and separates in train_val/test set foldlabels if requested
    if (config.foldlabel == True) and (config.mode != 'evaluation'):
        check_subject_consistency(config.subjects_all,
                                  config.subjects_foldlabel_all)
        train_val_foldlabel_subjects, train_val_foldlabel_data, \
            test_foldlabel_subjects, test_foldlabel_data = \
            extract_data(config.foldlabel_all, config)
        log.info("foldlabel data loaded")

        # Makes some sanity checks
        check_if_same_subjects(train_val_subjects,
                               train_val_foldlabel_subjects, "train_val")
        check_if_same_subjects(test_subjects,
                               test_foldlabel_subjects, "test")
        check_if_same_shape(train_val_data,
                            train_val_foldlabel_data, "train_val")
        check_if_same_shape(test_data,
                            test_foldlabel_data, "test")
    else:
        log.info("foldlabel data NOT requested. Foldlabel data NOT loaded")

    # Creates the dataset from these data by doing some preprocessing
    if config.mode == 'evaluation':
        test_dataset = ContrastiveDataset_Visualization(
            filenames=test_subjects,
            array=test_data,
            config=config)
        train_val_dataset = ContrastiveDataset_Visualization(
            filenames=train_val_subjects,
            array=train_val_data,
            config=config)
    else:
        if config.foldlabel == True:
            test_dataset = ContrastiveDataset_WithFoldLabels(
                filenames=test_subjects,
                array=test_data,
                foldlabel_array=test_foldlabel_data,
                config=config)
            train_val_dataset = ContrastiveDataset_WithFoldLabels(
                filenames=train_val_subjects,
                array=train_val_data,
                foldlabel_array=train_val_foldlabel_data,
                config=config)
        else:
            test_dataset = ContrastiveDataset(
                filenames=test_subjects,
                array=test_data,
                config=config)
            train_val_dataset = ContrastiveDataset(
                filenames=train_val_subjects,
                array=train_val_data,
                config=config)

    train_dataset, val_dataset = \
        extract_train_val_dataset(train_val_dataset,
                                  config.partition,
                                  config.seed)
    
    # just to have the same data format as train and val
    test_dataset, _ = torch.utils.data.random_split(
        test_dataset,
        [len(test_dataset),0])

    return train_dataset, val_dataset, test_dataset, train_val_dataset


def create_sets_with_labels(config):
    """Creates train, validation and test sets when there are labels

    Args:
        config (Omegaconf dict): contains configuration parameters
    Returns:
        train_dataset, val_dataset, test_datasetset, train_val_dataset (tuple)
    """

    # Gets labels for all subjects
    # Column subject_column_name is renamed 'Subject'
    subject_labels = read_labels(config.subject_labels_file,
                                 config.subject_column_name,
                                 config.label_names)

    if config.environment == "brainvisa" and config.checking:
        compare_npy_file_aims_files(config.subjects_all, config.numpy_all, config.crop_dir)

    # Loads and separates in train_val/test skeleton crops
    train_val_subjects, train_val_data, train_val_labels,\
    test_subjects, test_data, test_labels = \
        extract_data_with_labels(config.numpy_all, subject_labels, config.crop_dir, config)

    check_if_skeleton(train_val_data, "train_val")
    check_if_skeleton(test_data, "test")

    if config.environment == "brainvisa" and config.checking:
        compare_array_aims_files(train_val_subjects, train_val_data, config.crop_dir)
        compare_array_aims_files(test_subjects, test_data, config.crop_dir)
    

    # Makes some sanity checks on ordering of label subjects
    check_if_same_subjects(train_val_subjects,
                           train_val_labels[['Subject']], "train_val labels")
    check_if_same_subjects(test_subjects,
                           test_labels[['Subject']], "test labels")

    # Loads and separates in train_val/test set foldlabels if requested
    if (config.foldlabel == True) and (config.mode != 'evaluation'):
        check_subject_consistency(config.subjects_all,
                                  config.subjects_foldlabel_all)
        train_val_foldlabel_subjects, train_val_foldlabel_data, \
        train_val_labels, test_foldlabel_subjects, \
        test_foldlabel_data, test_labels = \
            extract_data_with_labels(config.foldlabel_all, subject_labels,
                                     config.foldlabel_dir, config)
        log.info("foldlabel data loaded")

        # Makes some sanity checks
        check_if_same_subjects(train_val_subjects,
                               train_val_foldlabel_subjects, "train_val")
        check_if_same_subjects(test_subjects,
                               test_foldlabel_subjects, "test")
        check_if_same_shape(train_val_data,
                            train_val_foldlabel_data, "train_val")
        check_if_same_shape(test_data,
                            test_foldlabel_data, "test")
        check_if_same_subjects(train_val_foldlabel_subjects,
                            train_val_labels[['Subject']], "train_val labels")
        check_if_same_subjects(test_foldlabel_subjects,
                            test_labels[['Subject']], "test labels")
        if config.environment == "brainvisa" and config.checking:
            compare_array_aims_files(train_val_foldlabel_subjects,
                                    train_val_foldlabel_data, config.foldlabel_dir)
            compare_array_aims_files(test_foldlabel_subjects,
                                    test_foldlabel_data, config.foldlabel_dir)
    else:
        log.info("foldlabel data NOT requested. Foldlabel data NOT loaded")

    # Creates the dataset from these data by doing some preprocessing
    if config.mode == 'evaluation':
        test_dataset = ContrastiveDataset_Visualization(
            filenames=test_subjects,
            array=test_data,
            config=config)
        train_val_dataset = ContrastiveDataset_Visualization(
            filenames=train_val_subjects,
            array=train_val_data,
            config=config)
    else:
        if config.foldlabel == True:
            test_dataset = ContrastiveDataset_WithLabels_WithFoldLabels(
                filenames=test_subjects,
                array=test_data,
                labels=test_labels,
                foldlabel_array=test_foldlabel_data,
                config=config)
            train_val_dataset = ContrastiveDataset_WithLabels_WithFoldLabels(
                filenames=train_val_subjects,
                array=train_val_data,
                labels=train_val_labels,
                foldlabel_array=train_val_foldlabel_data,
                config=config)
        else:
            test_dataset = ContrastiveDataset_WithLabels(
                filenames=test_subjects,
                array=test_data,
                labels=test_labels,
                config=config)
            train_val_dataset = ContrastiveDataset_WithLabels(
                filenames=train_val_subjects,
                array=train_val_data,
                labels=train_val_labels,
                config=config)

    train_dataset, val_dataset = \
        extract_train_val_dataset(train_val_dataset,
                                  config.partition,
                                  config.seed)

    return train_dataset, val_dataset, test_dataset, train_val_dataset
