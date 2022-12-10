# /usr/bin/env python3
# coding: utf-8
#

import os
import sys

import numpy as np
import pandas as pd
import json
import itertools
import torch

from train import train_vae
from load_data import create_subset
from config import Config


if __name__ == '__main__':

    config = Config()

    torch.manual_seed(5)
    save_dir = config.save_dir

    """ Load data and generate torch datasets """
    subset1 = create_subset(config)
    train_set, val_set = torch.utils.data.random_split(subset1,
                            [round(0.8*len(subset1)), round(0.2*len(subset1))])
    trainloader = torch.utils.data.DataLoader(
                  train_set,
                  batch_size=config.batch_size,
                  num_workers=8,
                  shuffle=True)
    valloader = torch.utils.data.DataLoader(
                val_set,
                batch_size=1,
                num_workers=8,
                shuffle=True)

    val_label = []
    for _, path in valloader:
        val_label.append(path[0])
    np.savetxt(f"{save_dir}val_label.csv", np.array(val_label), delimiter =", ", fmt ='% s')

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"

    weights = [1, 2]
    class_weights = torch.FloatTensor(weights).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights, reduction='sum')


    cur_config = {"kl": config.kl, "n": config.n}
    try:
        os.mkdir(save_dir)
    except FileExistsError:
        print("Directory " , save_dir ,  " already exists")
        pass
    print(cur_config)

    """ Train model for given configuration """
    vae, final_loss_val = train_vae(config, trainloader, valloader,
                                    root_dir=save_dir)

    # """ Evaluate model performances """
    # dico_set_loaders = {'train': trainloader, 'val': valloader}
    #
    # tester = ModelTester(model=vae, dico_set_loaders=dico_set_loaders,
    #                      kl_weight=config.kl, loss_func=criterion,
    #                      n_latent=config.n, depth=3)
    #
    # results = tester.test()
    # encoded = {loader_name:[results[loader_name][k] for k in results[loader_name].keys()] for loader_name in dico_set_loaders.keys()}
    # df_encoded = pd.DataFrame()
    # df_encoded['latent'] = encoded['train'] + encoded['val']
    # X = np.array(list(df_encoded['latent']))
    #
    # cluster = Cluster(X, save_dir)
    # res = cluster.plot_silhouette()
    # res['loss_val'] = final_loss_val
    #
    # with open(f"{save_dir}results_test.json", "w") as json_file:
    #     json_file.write(json.dumps(res, sort_keys=True, indent=4))
