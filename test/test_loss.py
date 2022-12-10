#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Test of loss functions and other relevant functions

"""

import torch
import numpy as np
from contrastive.losses import GeneralizedSupervisedNTXenLoss
from contrastive.utils import logs

log = logs.set_file_logger(__file__)

def test_mock():
    loss = GeneralizedSupervisedNTXenLoss()

def test_weights_two_labels():
    """Verify weights in a simple setting"""
    loss = GeneralizedSupervisedNTXenLoss(temperature=1.0, sigma=0.1)
    z_i = torch.Tensor([[0], 
                        [1]]) # not used in test
    z_j = torch.Tensor([[1], 
                        [0]])  # not used in test
    labels = torch.Tensor([[0], 
                           [1]])
    weights_ref = torch.tensor([[0, 0, 1, 0],
                                [0, 0, 0, 1],
                                [1, 0, 0, 0],
                                [0, 1, 0, 0]])
    _, weights = loss.forward_supervised(z_i, z_j, labels)
    assert torch.allclose(weights.double(), weights_ref.double())


def test_weights_one_label():
    """Verify weights in a simple setting"""
    loss = GeneralizedSupervisedNTXenLoss(temperature=1.0, sigma=0.1)
    z_i = torch.Tensor([[0], 
                        [1]]) # not used in test
    z_j = torch.Tensor([[1], 
                        [0]])  # not used in test
    labels = torch.Tensor([[0], 
                           [0]])
    weights_ref = 1/3.* torch.tensor([[0, 1, 1, 1],
                                      [1, 0, 1, 1],
                                      [1, 1, 0, 1],
                                      [1, 1, 1, 0]])
    _, weights = loss.forward_supervised(z_i, z_j, labels)
    assert torch.allclose(weights.double(), weights_ref.double())


def test_supervised_all_equal():
    """Verify loss labels in a simple setting.
    
    All vectors are equal"""
    loss = GeneralizedSupervisedNTXenLoss(temperature=1.0, sigma=0.1)
    z_i = torch.Tensor([[1, 0], 
                        [1, 0]]) # [N,D]]
    z_j = torch.Tensor([[1, 0], 
                        [1, 0]]) # [N,D]]
    labels = torch.Tensor([[0], 
                           [1]]) # [N]
    
    loss_label, _ = loss.forward_supervised(z_i, z_j, labels)
    loss_label_ref = 2*np.log(3.)
    loss_label_ref = torch.from_numpy(np.array((loss_label_ref)))
    assert torch.allclose(loss_label.double(),
                          loss_label_ref.double())


def test_supervised_different_for_different_labels():
    """Verify weights in a simple setting
    
    All vectors beloning to same label are equal"""
    loss = GeneralizedSupervisedNTXenLoss(temperature=1.0, sigma=0.1)
    z_i = torch.Tensor([[1, 0], 
                        [0, 1]]) # [N,D]]
    z_j = torch.Tensor([[1, 0], 
                        [0, 1]]) # [N,D]]
    labels = torch.Tensor([[0], 
                           [1]]) # [N]
    
    loss_label, _ = loss.forward_supervised(z_i, z_j, labels)
    loss_label_ref = 2*np.log(np.exp(1)+2) - 2
    loss_label_ref = torch.from_numpy(np.array((loss_label_ref)))
    assert torch.allclose(loss_label.double(),
                          loss_label_ref.double())


def test_supervised_3_labels_all_equal():
    """Verify loss labels in a simple setting.
    
    3 vectors, 2 labels, all vectors are equal"""
    loss = GeneralizedSupervisedNTXenLoss(temperature=1.0, sigma=0.1)
    z_i = torch.Tensor([[1, 0], 
                        [1, 0],
                        [1, 0]]) # [N,D]]
    z_j = torch.Tensor([[1, 0], 
                        [1, 0],
                        [1, 0]]) # [N,D]]
    labels = torch.Tensor([[0], 
                           [1],
                           [1]]) # [N]
    
    loss_label, _ = loss.forward_supervised(z_i, z_j, labels)
    loss_label_ref = 2*np.log(5.)
    loss_label_ref = torch.from_numpy(np.array((loss_label_ref)))
    assert torch.allclose(loss_label.double(),
                          loss_label_ref.double())


def test_supervised_3_labels_all_equal_different_for_different_labels():
    """Verify loss labels in a simple setting.
    
    3 vectors, 2 labels, all vectors for same are equal"""
    loss = GeneralizedSupervisedNTXenLoss(temperature=1.0, sigma=0.1)
    z_i = torch.Tensor([[0, 1], 
                        [1, 0],
                        [1, 0]]) # [N,D]]
    z_j = torch.Tensor([[0, 1], 
                        [1, 0],
                        [1, 0]]) # [N,D]]
    labels = torch.Tensor([[0], 
                           [1],
                           [1]]) # [N]
    
    loss_label, _ = loss.forward_supervised(z_i, z_j, labels)
    loss_label_ref = -2.0 + \
                     2./3*np.log(np.exp(1)+4) + \
                     4./3*np.log(3*np.exp(1)+2)
    loss_label_ref = torch.from_numpy(np.array((loss_label_ref)))
    assert torch.allclose(loss_label.double(),
                          loss_label_ref.double())


def test_pure_contrastive():
    """Verifies pure contrastive results in a simple setting"""
    loss = GeneralizedSupervisedNTXenLoss(temperature=1.0, sigma=0.1)
    z_i = torch.Tensor([[0], 
                        [1]]) # N=2; D=1
    z_j = torch.Tensor([[1], 
                        [0]])  # N=2, D=1
    loss_contrastive = loss.forward_pure_contrastive(z_i, z_j)
    loss_contrastive_ref = np.log(3.) + np.log(2+np.exp(1))
    loss_contrastive_ref = torch.from_numpy(np.array(loss_contrastive_ref))
    assert torch.allclose(loss_contrastive.double(),
                          loss_contrastive_ref.double())


def test_pure_contrastive_D_2_different_positive():
    """Verifies pure contrastive results in a simple setting
    
    When the two positive pairs are orthogonal"""
    loss = GeneralizedSupervisedNTXenLoss(temperature=1.0, sigma=0.1)
    z_i = torch.Tensor([[0, 1], 
                        [1, 0]]) # N=2; D=2
    z_j = torch.Tensor([[1, 0], 
                        [0, 1]])  # N=2, D=1
    loss_contrastive = loss.forward_pure_contrastive(z_i, z_j)
    loss_contrastive_ref = 2*np.log(2+np.exp(1))
    loss_contrastive_ref = torch.from_numpy(np.array(loss_contrastive_ref))
    assert torch.allclose(loss_contrastive.double(),
                          loss_contrastive_ref.double())


def test_pure_contrastive_D_2_equal_positive():
    """Verifies pure contrastive results in a simple setting
    
    When the two positive pairs are equal"""
    loss = GeneralizedSupervisedNTXenLoss(temperature=1.0, sigma=0.1)
    z_i = torch.Tensor([[0, 1], 
                        [1, 0]]) # N=2; D=2
    z_j = torch.Tensor([[0, 1], 
                        [1, 0]])  # N=2, D=1
    loss_contrastive = loss.forward_pure_contrastive(z_i, z_j)
    loss_contrastive_ref = 2*np.log(2+np.exp(1)) - 2.0
    loss_contrastive_ref = torch.from_numpy(np.array(loss_contrastive_ref))
    assert torch.allclose(loss_contrastive.double(),
                          loss_contrastive_ref.double())


def test_pure_contrastive_all_ones():
    """Verifies pure contrastive results in a simple setting
    
    All vectors are equal"""
    loss = GeneralizedSupervisedNTXenLoss(temperature=1.0, sigma=0.1)
    z_i = torch.Tensor([[1], 
                        [1]]) # N=2; D=1
    z_j = torch.Tensor([[1], 
                        [1]])  # N=2, D=1
    loss_contrastive = loss.forward_pure_contrastive(z_i, z_j)
    loss_contrastive_ref = 2*np.log(3.)
    loss_contrastive_ref = torch.from_numpy(np.array(loss_contrastive_ref))
    assert torch.allclose(loss_contrastive.double(),
                          loss_contrastive_ref.double())


def test_compare_supervised_unsupervised():
    """Compare supervised and pure contrastive losses
    
    When all labels are different, 
    both supervised and contrastive losses should be equal"""
    loss = GeneralizedSupervisedNTXenLoss(temperature=1.0, sigma=0.1)
    z_i = torch.randint(0, 20, (10,3)).float()
    z_j = torch.randint(0, 20, (10,3)).float()
    labels = torch.arange(0,10).T
    loss_labels, _ = loss.forward_supervised(z_i, z_j, labels)
    loss_pure_contrastive = loss.forward_pure_contrastive(z_i, z_j)
    assert torch.allclose(loss_labels.double(), loss_pure_contrastive.double())


def test_forward_supervised_D_2():
    loss = GeneralizedSupervisedNTXenLoss(temperature=1.0, sigma=0.1)


def test_forward_supervised():
    loss = GeneralizedSupervisedNTXenLoss(temperature=1.0, sigma=0.1)


if __name__ == "__main__":
    test_compare_supervised_unsupervised()