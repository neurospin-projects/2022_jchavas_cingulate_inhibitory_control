# -*- coding: utf-8 -*-
# /usr/bin/env python3
#
import matplotlib.pyplot as plt


def plot_loss(list_loss_train, root_dir, *list_loss_val):
    """
    Plot training loss given two lists of loss
    Args:
        list_loss_train: list of loss values of training set
        list_loss_val: list of validation set loss values
    """
    plt.clf()
    plt.subplot()
    epoch = [k for k in range(1, len(list_loss_train) + 1)]
    plt.plot(epoch, list_loss_train, label='Train')
    if list_loss_val:
        plt.plot(epoch, list_loss_val, label='Validation')
    plt.xlabel('Number of epochs')
    plt.ylabel('Loss value')
    plt.legend()
    plt.savefig(root_dir+"loss.png")
