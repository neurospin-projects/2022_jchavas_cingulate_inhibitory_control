#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Data module
"""
import pytorch_lightning as pl
import numpy as np
import time
from torch import Generator
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler

from contrastive.data.create_datasets import create_sets_with_labels

from contrastive.data.create_datasets import create_sets_without_labels


class DataModule(pl.LightningDataModule):
    """Parent data module class
    """

    def __init__(self, config):
        super(DataModule, self).__init__()
        self.config = config

    def setup(self, stage=None, mode=None):
        if self.config.with_labels == True:
            self.dataset_train, self.dataset_val, self.dataset_test, self.dataset_train_val = \
                create_sets_with_labels(self.config)
        else:
            self.dataset_train, self.dataset_val, self.dataset_test, self.dataset_train_val = \
                create_sets_without_labels(self.config)


class DataModule_Learning(DataModule):
    """Data module class for Learning
    """

    def __init__(self, config):
        super(DataModule_Learning, self).__init__(config)
        self.gen = Generator()

    def train_dataloader(self):
        loader_train = DataLoader(self.dataset_train,
                                  batch_size=self.config.batch_size,
                                  sampler=RandomSampler(
                                      data_source=self.dataset_train,
                                      generator=self.gen),
                                  pin_memory=self.config.pin_mem,
                                  multiprocessing_context='fork',
                                  num_workers=self.config.num_cpu_workers
                                #   worker_init_fn = lambda _: np.random.seed(int(time.time()))
                                  )
        return loader_train

    def val_dataloader(self):
        loader_val = DataLoader(self.dataset_val,
                                batch_size=self.config.batch_size,
                                pin_memory=self.config.pin_mem,
                                multiprocessing_context='fork',
                                num_workers=self.config.num_cpu_workers,
                                shuffle=False
                                )
        return loader_val

    def test_dataloader(self):
        loader_test = DataLoader(self.dataset_test,
                                 batch_size=self.config.batch_size,
                                 pin_memory=self.config.pin_mem,
                                 multiprocessing_context='fork',
                                 num_workers=self.config.num_cpu_workers,
                                 shuffle=False
                                 )
        return loader_test

class DataModule_Evaluation(DataModule):
    """Data module class for evaluation/visualization
    """

    def __init__(self, config):
        super(DataModule_Evaluation, self).__init__(config)

    def train_val_dataloader(self):
        loader_train_val = DataLoader(self.dataset_train_val,
                                      batch_size=self.config.batch_size,
                                      pin_memory=self.config.pin_mem,
                                      num_workers=self.config.num_cpu_workers,
                                      shuffle=False
                                      )
        return loader_train_val

    def train_dataloader(self):
        loader_train = DataLoader(self.dataset_train,
                                  batch_size=self.config.batch_size,
                                  pin_memory=self.config.pin_mem,
                                  num_workers=self.config.num_cpu_workers,
                                  shuffle=False
                                  )
        return loader_train

    def val_dataloader(self):
        loader_val = DataLoader(self.dataset_val,
                                batch_size=self.config.batch_size,
                                pin_memory=self.config.pin_mem,
                                num_workers=self.config.num_cpu_workers,
                                shuffle=False
                                )
        return loader_val

    def test_dataloader(self):
        loader_test = DataLoader(self.dataset_test,
                                 batch_size=self.config.batch_size,
                                 pin_memory=self.config.pin_mem,
                                 num_workers=self.config.num_cpu_workers,
                                 shuffle=False
                                 )
        return loader_test