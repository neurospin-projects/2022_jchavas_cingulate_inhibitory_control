#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A test to just analyze randomly generated input images
"""
import torch
import numpy as np

from contrastive.losses import NTXenLoss
from contrastive.models.contrastive_learner import ContrastiveLearner

try:
    from contrastive.utils.plots.visualize_anatomist import Visu_Anatomist
except ImportError:
    print("INFO: you are probably not in a brainvisa environment. Probably OK.")


class ContrastiveLearner_Visualization(ContrastiveLearner):

    def __init__(self, config, sample_data):
        super(ContrastiveLearner_Visualization, self).__init__(
            config=config, sample_data=sample_data)
        self.config = config
        self.sample_data = sample_data
        self.sample_i = np.array([])
        self.sample_j = np.array([])
        self.val_sample_i = np.array([])
        self.val_sample_j = np.array([])
        self.recording_done = False
        if self.config.environment == "brainvisa":
            self.visu_anatomist = Visu_Anatomist()
        self.eval()

    def custom_histogram_adder(self):

        # iterating through all parameters
        for name, params in self.named_parameters():

            self.logger.experiment.add_histogram(
                name, params, self.current_epoch)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.config.lr,
                                     weight_decay=self.config.weight_decay)
        return optimizer

    def nt_xen_loss(self, z_i, z_j):
        loss = NTXenLoss(temperature=self.config.temperature,
                         return_logits=True)
        return loss.forward(z_i, z_j)

    def training_step(self, train_batch, batch_idx):
        (inputs, filenames) = train_batch
        if batch_idx == 0:
            self.sample_i.append(inputs[:, 0, :].cpu())
            self.sample_j.append(inputs[:, 1, :].cpu())

    def training_epoch_end(self, outputs):
        if self.config.environment == "brainvisa":
            image_input_i = self.visu_anatomist.plot_bucket(
                self.sample_i, buffer=True)
            self.logger.experiment.add_image(
                'input_test_i', image_input_i, self.current_epoch)
            image_input_j = self.visu_anatomist.plot_bucket(
                self.sample_j, buffer=True)
            self.logger.experiment.add_image(
                'input_test_j', image_input_j, self.current_epoch)
