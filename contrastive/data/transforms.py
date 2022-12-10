#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Transforms used in dataset
"""

import torchvision.transforms as transforms

from contrastive.augmentations import *


def transform_nothing_done():
    return \
        transforms.Compose([
            SimplifyTensor(),
            EndTensor()
        ])

def transform_only_padding(config):
    if config.backbone_name != 'pointnet':
        return \
            transforms.Compose([
                SimplifyTensor(),
                PaddingTensor(shape=config.input_size,
                            fill_value=config.fill_value),
                BinarizeTensor(),
                EndTensor()
            ])
    else:
        return \
            transforms.Compose([
                SimplifyTensor(),
                PaddingTensor(shape=config.input_size,
                            fill_value=config.fill_value),
                BinarizeTensor(),
                EndTensor(),
                ToPointnetTensor(n_max=config.n_max)
            ])


def transform_foldlabel(sample_foldlabel, percentage, config):
    if config.backbone_name != 'pointnet':
        return \
            transforms.Compose([
                SimplifyTensor(),
                PaddingTensor(shape=config.input_size,
                            fill_value=config.fill_value),
                RemoveRandomBranchTensor(sample_foldlabel=sample_foldlabel,
                                        percentage=percentage,
                                        variable_percentage = config.variable_percentage,
                                        input_size=config.input_size,
                                        keep_bottom=config.keep_bottom),
                RotateTensor(max_angle=config.max_angle),
                BinarizeTensor()
            ])
    else:
        return \
            transforms.Compose([
                SimplifyTensor(),
                PaddingTensor(shape=config.input_size,
                            fill_value=config.fill_value),
                RemoveRandomBranchTensor(sample_foldlabel=sample_foldlabel,
                                        percentage=percentage,
                                        variable_percentage = config.variable_percentage,
                                        input_size=config.input_size,
                                        keep_bottom=config.keep_bottom),
                RotateTensor(max_angle=config.max_angle),
                BinarizeTensor(),
                ToPointnetTensor(n_max=config.n_max)
            ])


def transform_no_foldlabel(from_skeleton, config):
    if config.backbone_name != 'pointnet':
        return \
            transforms.Compose([
                SimplifyTensor(),
                PaddingTensor(shape=config.input_size,
                            fill_value=config.fill_value),
                PartialCutOutTensor_Roll(from_skeleton=from_skeleton,
                                        keep_bottom=config.keep_bottom,
                                        patch_size=config.patch_size),
                RotateTensor(max_angle=config.max_angle),
                BinarizeTensor()
            ])
    else:
        return \
            transforms.Compose([
                SimplifyTensor(),
                PaddingTensor(shape=config.input_size,
                            fill_value=config.fill_value),
                PartialCutOutTensor_Roll(from_skeleton=from_skeleton,
                                        keep_bottom=config.keep_bottom,
                                        patch_size=config.patch_size),
                RotateTensor(max_angle=config.max_angle),
                BinarizeTensor(),
                ToPointnetTensor(n_max=config.n_max)
            ])