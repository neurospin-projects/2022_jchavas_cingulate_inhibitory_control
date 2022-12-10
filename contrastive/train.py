#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Training contrastive on skeleton images

"""
######################################################################
# Imports and global variables definitions
######################################################################
import os

import hydra
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

from contrastive.data.datamodule import DataModule_Learning
from contrastive.data.datamodule import DataModule_Evaluation
from contrastive.models.contrastive_learner import ContrastiveLearner
from contrastive.models.contrastive_learner_with_labels import \
    ContrastiveLearner_WithLabels
from contrastive.models.contrastive_learner_visualization import \
    ContrastiveLearner_Visualization
from contrastive.utils.config import create_accessible_config, process_config,\
get_config_diff
from contrastive.utils.logs import set_root_logger_level
from contrastive.utils.logs import set_file_log_handler
from contrastive.utils.logs import set_file_logger

tb_logger = pl_loggers.TensorBoardLogger('logs')
writer = SummaryWriter()
log = set_file_logger(__file__)

"""
We use the following definitions:
- embedding or representation, the space before the projection head.
  The elements of the space are features
- output, the space after the projection head.
  The elements are called output vectors
"""

@hydra.main(config_name='config', config_path="configs")
def train(config):
    config = process_config(config)
    os.environ["NUMEXPR_MAX_THREADS"] = str(config.num_cpu_workers)

    set_root_logger_level(config.verbose)
    # Sets handler for logger
    set_file_log_handler(file_dir=os.getcwd(),
                         suffix='output')
    log.info(f"current directory = {os.getcwd()}")


    # copies some of the config parameters in a yaml file easily accessible
    keys_to_keep = ['dataset_name', 'nb_subjects', 'model', 'with_labels', 
    'input_size', 'temperature_initial', 'temperature', 'sigma', 'drop_rate', 'depth_decoder',
    'mode', 'foldlabel', 'fill_value', 'patch_size', 'max_angle', 'checkerboard_size', 'keep_bottom',
    'growth_rate', 'block_config', 'num_init_features', 'num_representation_features', 'num_outputs',
    'environment', 'batch_size', 'pin_mem', 'partition', 'lr', 'weight_decay', 'max_epochs',
    'early_stopping_patience', 'seed', 'backbone_name', 'sigma_labels', 'proportion_pure_contrastive', 'n_max',
    'train_val_csv_file']
    if config.model == 'SimCLR_supervised':
        keys_to_keep.extend(['temperature_supervised', 'sigma_labels', 'pretrained_model_path'])

    create_accessible_config(keys_to_keep, os.getcwd()+"/.hydra/config.yaml")

    # create a csv file where the parameters changing between runs are stored
    get_config_diff(os.getcwd()+'/..', whole_config=False, save=True)    


    if config.mode == 'evaluation':
        data_module = DataModule_Evaluation(config)
    else:
        data_module = DataModule_Learning(config)

    if config.mode == 'evaluation':
        model = ContrastiveLearner_Visualization(config,
                               sample_data=data_module)   
    elif config.model == "SimCLR_supervised":
        model = ContrastiveLearner_WithLabels(config,
                               sample_data=data_module)
    elif config.model == 'SimCLR':
        model = ContrastiveLearner(config,
                               sample_data=data_module) 
    else:
        raise ValueError("Wrong combination of 'mode' and 'model'")


    if config.backbone_name != 'pointnet':
        summary(model, tuple(config.input_size), device="cpu")
    else:
        summary(model, device='cpu')

    early_stop_callback = EarlyStopping(monitor="val_loss",
          patience=config.early_stopping_patience)

    trainer = pl.Trainer(
        gpus=1,
        max_epochs=config.max_epochs,
        callbacks=[early_stop_callback],
        logger=tb_logger,
        flush_logs_every_n_steps=config.nb_steps_per_flush_logs,
        log_every_n_steps=config.log_every_n_steps)

    trainer.fit(model, data_module, ckpt_path=config.checkpoint_path)
    log.info("Fitting is done")
    log.info(f"Number of hooks: {len(model.save_output.outputs)} ; {len(model.hook_handles)}")


if __name__ == "__main__":
    train()
