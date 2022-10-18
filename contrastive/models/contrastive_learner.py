#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  This software and supporting documentation are distributed by
#      Institut Federatif de Recherche 49
#      CEA/NeuroSpin, Batiment 145,
#      91191 Gif-sur-Yvette cedex
#      France
#
# This software is governed by the CeCILL license version 2 under
# French law and abiding by the rules of distribution of free software.
# You can  use, modify and/or redistribute the software under the
# terms of the CeCILL license version 2 as circulated by CEA, CNRS
# and INRIA at the following URL "http://www.cecill.info".
#
# As a counterpart to the access to the source code and  rights to copy,
# modify and redistribute granted by the license, users are provided only
# with a limited warranty  and the software's author,  the holder of the
# economic rights,  and the successive licensors  have only  limited
# liability.
#
# In this respect, the user's attention is drawn to the risks associated
# with loading,  using,  modifying and/or developing or reproducing the
# software by the user in light of its specific status of free software,
# that may mean  that it is complicated to manipulate,  and  that  also
# therefore means  that it is reserved for developers  and  experienced
# professionals having in-depth computer knowledge. Users are therefore
# encouraged to load and test the software's suitability as regards their
# requirements in conditions enabling the security of their systems and/or
# data to be ensured and,  more generally, to use and operate it in the
# same conditions as regards security.
#
# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license version 2 and that you accept its terms.
"""
Some helper functions are taken from:
https://learnopencv.com/tensorboard-with-pytorch-lightning

"""
import numpy as np
import torch
import pytorch_lightning as pl
from sklearn.manifold import TSNE
from toolz.itertoolz import first
from contrastive.augmentations import ToPointnetTensor

from contrastive.backbones.densenet import DenseNet
from contrastive.backbones.convnet import ConvNet
from contrastive.backbones.pointnet import PointNetCls
from contrastive.losses import NTXenLoss
from contrastive.losses import CrossEntropyLoss
from contrastive.utils.plots.visualize_images import plot_bucket
from contrastive.utils.plots.visualize_images import plot_histogram
from contrastive.utils.plots.visualize_images import plot_histogram_weights
from contrastive.utils.plots.visualize_images import plot_histogram
from contrastive.utils.plots.visualize_images import plot_scatter_matrix
from contrastive.utils.plots.visualize_tsne import plot_tsne

try:
    from contrastive.utils.plots.visualize_anatomist import Visu_Anatomist
except ImportError:
    print("INFO: you are probably not in a brainvisa environment. Probably OK.")

    
class SaveOutput:
    def __init__(self):
        self.outputs = {}

    def __call__(self, module, module_in, module_out):
        self.outputs[module] = module_out.cpu()

    def clear(self):
        self.outputs = {}


class ContrastiveLearner(pl.LightningModule):

    def __init__(self, config, sample_data):
        super(ContrastiveLearner, self).__init__()
        if config.backbone_name == 'densenet':
            self.backbone = DenseNet(
                growth_rate=config.growth_rate,
                block_config=config.block_config,
                num_init_features=config.num_init_features,
                num_representation_features=config.num_representation_features,
                num_outputs=config.num_representation_features,
                projection_head_type=config.projection_head_type,
                mode=config.mode,
                drop_rate=config.drop_rate,
                in_shape=config.input_size,
                depth=config.depth_decoder)
        elif config.backbone_name == "convnet":
            self.backbone = ConvNet(
                encoder_depth=config.encoder_depth,
                num_representation_features=config.num_representation_features,
                num_outputs=config.num_representation_features,
                projection_head_hidden_layers=config.projection_head_hidden_layers,
                projection_head_type=config.projection_head_type,
                drop_rate=config.drop_rate,
                mode=config.mode,
                in_shape=config.input_size)
        elif config.backbone_name == 'pointnet':
            self.backbone = PointNetCls(
                k=config.num_representation_features,
                num_outputs=config.num_representation_features,
                projection_head_hidden_layers=config.projection_head_hidden_layers,
                # projection_head_type=config.projection_head_type,
                drop_rate=config.drop_rate,
                feature_transform=False)
        self.config = config
        self.sample_data = sample_data
        self.sample_i = np.array([])
        self.sample_j = np.array([])
        self.sample_k = np.array([])
        self.sample_filenames = []
        self.save_output = SaveOutput()
        self.hook_handles = []
        self.get_layers()
        if self.config.environment == "brainvisa":
            self.visu_anatomist = Visu_Anatomist()


    def forward(self, x):
        return self.backbone.forward(x)

    def get_layers(self):
        i = 0
        for layer in self.modules():
            if self.config.backbone_name in ['densenet', 'convnet']:
                if isinstance(layer, torch.nn.BatchNorm1d):
                    handle = layer.register_forward_hook(self.save_output)
                    self.hook_handles.append(handle)
            elif self.config.backbone_name == 'pointnet':
                # for the moment, keep the same method
                # need to pass the wanted representation layer to the first place
                # => remove the first five layers
                if isinstance(layer, torch.nn.Linear):
                    if i >= 5:
                        handle = layer.register_forward_hook(self.save_output)
                        self.hook_handles.append(handle)
                    i += 1


    def custom_histogram_adder(self):
        """Builds histogram for each model parameter.
        """
        # iterating through all parameters
        for name, params in self.named_parameters():
            self.logger.experiment.add_histogram(
                name,
                params,
                self.current_epoch)

    def plot_histograms(self):
        """Plots all zii, zjj, zij and weights histograms"""
        # Computes histogram of sim_zii
        histogram_sim_zii = plot_histogram(self.sim_zii, buffer=True)
        self.logger.experiment.add_image(
            'histo_sim_zii', histogram_sim_zii, self.current_epoch)

        # Computes histogram of sim_zjj
        histogram_sim_zjj = plot_histogram(self.sim_zjj, buffer=True)
        self.logger.experiment.add_image(
            'histo_sim_zjj', histogram_sim_zjj, self.current_epoch)

        # Computes histogram of sim_zij
        histogram_sim_zij = plot_histogram(self.sim_zij, buffer=True)
        self.logger.experiment.add_image(
            'histo_sim_zij', histogram_sim_zij, self.current_epoch)

        # Computes histogram of weights
        histogram_weights = plot_histogram_weights(self.weights,
                                                    buffer=True)
        self.logger.experiment.add_image(
            'histo_weights', histogram_weights, self.current_epoch)

    def plot_scatter_matrices(self):
        """Plots scatter matrices of output and representations spaces"""
        # Makes scatter matrix of output space
        r = self.compute_outputs_skeletons(
            self.sample_data.train_dataloader())
        X = r[0] # First element of tuple
        scatter_matrix_outputs = plot_scatter_matrix(X, buffer=True)
        self.logger.experiment.add_image(
            'scatter_matrix_outputs',
            scatter_matrix_outputs,
            self.current_epoch)

        # Makes scatter matrix of representation space
        r = self.compute_representations(
            self.sample_data.train_dataloader())
        X = r[0] # First element of tuple
        scatter_matrix_representations = plot_scatter_matrix(
            X, buffer=True)
        self.logger.experiment.add_image(
            'scatter_matrix_representations',
            scatter_matrix_representations,
            self.current_epoch)

    def plot_views(self):
        """Plots different 3D views"""
        image_input_i = plot_bucket(self.sample_i, buffer=True)
        self.logger.experiment.add_image(
            'input_i', image_input_i, self.current_epoch)
        image_input_j = plot_bucket(self.sample_j, buffer=True)
        self.logger.experiment.add_image(
            'input_j', image_input_j, self.current_epoch)

        # Plots view using anatomist
        if self.config.environment == "brainvisa":
            image_input_i = self.visu_anatomist.plot_bucket(
                self.sample_i, buffer=True)
            self.logger.experiment.add_image(
                'input_ana_i: ',
                image_input_i, self.current_epoch)
            # self.logger.experiment.add_text(
            #     'filename: ',self.sample_filenames[0], self.current_epoch)
            image_input_j = self.visu_anatomist.plot_bucket(
                self.sample_j, buffer=True)
            self.logger.experiment.add_image(
                'input_ana_j: ',
                image_input_j, self.current_epoch)
            if len(self.sample_k) != 0:
                image_input_k = self.visu_anatomist.plot_bucket(
                    self.sample_k, buffer=True)
                self.logger.experiment.add_image(
                    'input_ana_k: ',
                    image_input_k, self.current_epoch)

    def configure_optimizers(self):
        """Adam optimizer"""
        optimizer = torch.optim.Adam(\
                        filter(lambda p: p.requires_grad, self.parameters()),
                        lr=self.config.lr,
                        weight_decay=self.config.weight_decay)
        return optimizer

    def nt_xen_loss(self, z_i, z_j):
        """Loss function for contrastive"""
        loss = NTXenLoss(temperature=self.config.temperature,
                         return_logits=True)
        return loss.forward(z_i, z_j)

    def cross_entropy_loss(self, sample, output_i, output_j):
        """Loss function for decoder"""
        loss = CrossEntropyLoss(device=self.device)
        return loss.forward(sample, output_i, output_j)

    def training_step(self, train_batch, batch_idx):
        """Training step.
        """
        (inputs, filenames) = train_batch
        if self.config.backbone_name == 'pointnet':
            inputs = torch.squeeze(inputs).to(torch.float)
        #print("TRAINING STEP", inputs.shape)
        input_i = inputs[:, 0, :]
        input_j = inputs[:, 1, :]
        z_i = self.forward(input_i)
        z_j = self.forward(input_j)

        if self.config.mode == "decoder":
            sample = inputs[:, 2, :]
            batch_loss = self.cross_entropy_loss(sample, z_i, z_j)
        else:
            batch_loss, sim_zij, sim_zii, sim_zjj = self.nt_xen_loss(z_i, z_j)

        self.log('train_loss', float(batch_loss))

        # Only computes graph on first step
        if self.global_step == 1:
            self.logger.experiment.add_graph(self, input_i)

        # Records sample for first batch of each epoch
        if batch_idx == 0:
            self.sample_i = input_i.cpu()
            self.sample_j = input_j.cpu()
            self.sample_filenames = filenames
            if self.config.mode != "decoder":
                self.sim_zij = sim_zij * self.config.temperature
                self.sim_zii = sim_zii * self.config.temperature
                self.sim_zjj = sim_zjj * self.config.temperature

        # logs - a dictionary
        logs = {"train_loss": float(batch_loss)}

        batch_dictionary = {
            # REQUIRED: It is required for us to return "loss"
            "loss": batch_loss,
            # optional for batch logging purposes
            "log": logs,
        }

        return batch_dictionary

    def compute_outputs_skeletons(self, loader):
        """Computes the outputs of the model for each crop.

        This includes the projection head"""

        # Initialization
        X = torch.zeros([0, self.config.num_representation_features]).cpu()
        filenames_list = []
        transform = ToPointnetTensor()

        # Computes embeddings without computing gradient
        with torch.no_grad():
            for (inputs, filenames) in loader:
                # First views of the whole batch
                inputs = inputs.cuda()
                model = self.cuda()
                input_i = inputs[:, 0, :]
                input_j = inputs[:, 1, :]
                if self.config.backbone_name == 'pointnet':
                    input_i = transform(input_i.cpu()).cuda().to(torch.float)
                    input_j = transform(input_j.cpu()).cuda().to(torch.float)
                X_i = model.forward(input_i)
                # Second views of the whole batch
                X_j = model.forward(input_j)
                # First views and second views
                # are put side by side
                X_reordered = torch.cat([X_i, X_j], dim=-1)
                X_reordered = X_reordered.view(-1, X_i.shape[-1])
                X = torch.cat((X, X_reordered.cpu()), dim=0)
                filenames_duplicate = [item
                                       for item in filenames
                                       for repetitions in range(2)]
                filenames_list = filenames_list + filenames_duplicate
                del inputs

        return X, filenames_list


    def compute_decoder_outputs_skeletons(self, loader):
        """Computes the outputs of the model for each crop.

        This includes the projection head"""

        # Initialization
        X = torch.zeros([0, 2, 20, 40, 40]).cpu()
        filenames_list = []

        # Computes embeddings without computing gradient
        with torch.no_grad():
            for (inputs, filenames) in loader:
                # First views of the whole batch
                inputs = inputs.cuda()
                model = self.cuda()
                X_i = model.forward(inputs[:, 0, :])
                print(f"shape X and X_i: {X.shape}, {X_i.shape}")
                # First views re put side by side
                X = torch.cat((X, X_i.cpu()), dim=0)
                filenames_duplicate = [item
                                       for item in filenames]
                filenames_list = filenames_list + filenames_duplicate
                del inputs

        return X, filenames_list

    def compute_representations(self, loader):
        """Computes representations for each crop.

        Representation are before the projection head"""

        # Initialization
        X = torch.zeros([0, self.config.num_representation_features]).cpu()
        filenames_list = []

        # Computes representation (without gradient computation)
        with torch.no_grad():
            for (inputs, filenames) in loader:
                # First views of the whole batch
                inputs = inputs.cuda()
                if self.config.backbone_name == 'pointnet':
                    inputs = torch.squeeze(inputs).to(torch.float)
                model = self.cuda()
                input_i = inputs[:, 0, :]
                input_j = inputs[:, 1, :]
                model.forward(input_i)
                X_i = first(self.save_output.outputs.values())
                # Second views of the whole batch
                model.forward(input_j)
                X_j = first(self.save_output.outputs.values())
                #print("representations", X_i.shape, X_j.shape)
                # First views and second views are put side by side
                X_reordered = torch.cat([X_i, X_j], dim=-1)
                X_reordered = X_reordered.view(-1, X_i.shape[-1])
                X = torch.cat((X, X_reordered.cpu()), dim=0)
                #print(f"filenames = {filenames}")
                filenames_duplicate = [
                    item for item in filenames
                    for repetitions in range(2)]
                filenames_list = filenames_list + filenames_duplicate
                del inputs

        return X, filenames_list

    def compute_tsne(self, loader, register):
        """Computes t-SNE.

        It is computed either in the representation
        or in the output space"""

        if register == "output":
            X, _ = self.compute_outputs_skeletons(loader)
        elif register == "representation":
            X, _ = self.compute_representations(loader)
        else:
            raise ValueError(
                "Argument register must be either output or representation")

        tsne = TSNE(n_components=2, perplexity=5, init='pca', random_state=50)

        Y = X.detach().numpy()

        # Makes the t-SNE fit
        X_tsne = tsne.fit_transform(Y)

        # Returns tsne embeddings
        return X_tsne

    def training_epoch_end(self, outputs):
        """Computation done at the end of the epoch"""

        if self.config.mode == "encoder":
            # Computes t-SNE both in representation and output space
            if self.current_epoch % self.config.nb_epochs_per_tSNE == 0 \
                    or self.current_epoch >= self.config.max_epochs:
                X_tsne = self.compute_tsne(
                    self.sample_data.train_dataloader(), "output")
                image_TSNE = plot_tsne(X_tsne, buffer=True)
                self.logger.experiment.add_image(
                    'TSNE output image', image_TSNE, self.current_epoch)
                X_tsne = self.compute_tsne(
                    self.sample_data.train_dataloader(), "representation")
                image_TSNE = plot_tsne(X_tsne, buffer=True)
                self.logger.experiment.add_image(
                    'TSNE representation image', image_TSNE, self.current_epoch)

            # Computes histogram of sim_zij
            histogram_sim_zij = plot_histogram(self.sim_zij, buffer=True)
            self.logger.experiment.add_image(
                'histo_sim_zij', histogram_sim_zij, self.current_epoch)

        # Plots views
        if self.config.backbone_name != 'pointnet':
            self.plot_views()

        # calculates average loss
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()

        # logs histograms
        self.custom_histogram_adder()

        # logging using tensorboard logger
        self.logger.experiment.add_scalar(
            "Loss/Train",
            avg_loss,
            self.current_epoch)

    def validation_step(self, val_batch, batch_idx):
        """Validation step"""

        (inputs, filenames) = val_batch
        if self.config.backbone_name == 'pointnet':
            inputs = torch.squeeze(inputs).to(torch.float)
        input_i = inputs[:, 0, :]
        input_j = inputs[:, 1, :]
        #print("INPUT I", input_i)
        z_i = self.forward(input_i)
        z_j = self.forward(input_j)

        if self.config.mode == "decoder":
            sample = inputs[:, 2, :]
            batch_loss = self.cross_entropy_loss(sample, z_i, z_j)
        else:
            batch_loss, sim_zij, sim_zii, sim_zjj = self.nt_xen_loss(z_i, z_j)
        self.log('val_loss', float(batch_loss))

        # logs- a dictionary
        logs = {"val_loss": float(batch_loss)}

        batch_dictionary = {
            # REQUIRED: It ie required for us to return "loss"
            "loss": batch_loss,

            # optional for batch logging purposes
            "log": logs,
        }

        return batch_dictionary

    def validation_epoch_end(self, outputs):
        """Computaion done at the end of each validation epoch"""

        # Computes t-SNE
        if self.config.mode == "encoder":
            if self.current_epoch % self.config.nb_epochs_per_tSNE == 0 \
                    or self.current_epoch >= self.config.max_epochs:
                X_tsne = self.compute_tsne(
                    self.sample_data.val_dataloader(), "output")
                image_TSNE = plot_tsne(X_tsne, buffer=True)
                self.logger.experiment.add_image(
                    'TSNE output validation image', image_TSNE, self.current_epoch)
                X_tsne = self.compute_tsne(
                    self.sample_data.val_dataloader(),
                    "representation")
                image_TSNE = plot_tsne(X_tsne, buffer=True)
                self.logger.experiment.add_image(
                    'TSNE representation validation image',
                    image_TSNE,
                    self.current_epoch)

        # calculates average loss
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()

        # logs losses using tensorboard logger
        self.logger.experiment.add_scalar(
            "Loss/Validation",
            avg_loss,
            self.current_epoch)
