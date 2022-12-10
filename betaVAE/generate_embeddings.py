# -*- coding: utf-8 -*-
# /usr/bin/env python3
#

import os
import pandas as pd
import torch

from beta_vae import VAE, ModelTester
from load_data import create_test_subset
from config import Config


def generate_emebdding_sets(embedding, config):
    """
    From a dataframe of encoded subjects, generate csv files:
    full_embeddings.csv, train_embeddings.csv etc.
    """
    data_dir = '/path/to/data'
    save_dir = config.test_model_dir

    # Loading of data subsets
    full = pd.read_csv(os.path.join(data_dir, 'full_subjects.csv'),
                header=None, usecols=[1], names=['ID'])
    full['ID'] = full['ID'].astype('str')

    train = pd.read_csv(os.path.join(data_dir, 'train_subjects.csv'),
                header=None, usecols=[1], names=['ID'])
    train['ID'] = train['ID'].astype('str')

    val = pd.read_csv(os.path.join(data_dir, 'val_subjects.csv'),
                header=None, usecols=[1], names=['ID'])
    val['ID'] = val['ID'].astype('str')

    test = pd.read_csv(os.path.join(data_dir, 'test_subjects.csv'),
                header=None, usecols=[1], names=['ID'])
    test['ID'] = test['ID'].astype('str')

    embedding['ID'] = embedding["ID"].astype('str')

    # Split of full ACC dataset embeddings as different subsets
    full_embeddings = embedding.merge(full, left_on='ID', right_on='ID')
    train_embeddings = embedding.merge(train, left_on='ID', right_on='ID')
    val_embeddings = embedding.merge(val, left_on='ID', right_on='ID')
    test_embeddings = embedding.merge(test, left_on='ID', right_on='ID')

    # Saving of ACC subsets as csv files
    full_embeddings.to_csv(os.path.join(save_dir, 'full_embeddings.csv'), index=False)
    train_embeddings.to_csv(os.path.join(save_dir, 'train_embeddings.csv'), index=False)
    val_embeddings.to_csv(os.path.join(save_dir, 'val_embeddings.csv'), index=False)
    test_embeddings.to_csv(os.path.join(save_dir, 'test_embeddings.csv'), index=False)


def main():
    """
    Infer a trained model on test data and saves the embeddings as csv
    """
    config = Config()

    torch.manual_seed(0)
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            vae = nn.DataParallel(vae)

    model_dir = os.path.join(config.test_model_dir, 'checkpoint.pt')
    model = VAE(config.in_shape, config.n, depth=3)
    model.load_state_dict(torch.load(model_dir))
    model = model.to(device)

    weights = [1, 2]
    class_weights = torch.FloatTensor(weights).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights, reduction='sum')

    subset_test = create_test_subset(config)
    testloader = torch.utils.data.DataLoader(
              subset_test,
              batch_size=config.batch_size,
              num_workers=8,
              shuffle=True)
    dico_set_loaders = {'test': testloader}

    tester = ModelTester(model=model, dico_set_loaders=dico_set_loaders,
                         kl_weight=config.kl, loss_func=criterion,
                         n_latent=config.n, depth=3)

    results = tester.test()
    embedding = pd.DataFrame(results['test']).T.reset_index()
    embedding = embedding.rename(columns={"index":"ID"})
    embedding = embedding.rename(columns={k:f"dim{k+1}" for k in range(config.n)})
    print(embedding.head())

    generate_emebdding_sets(embedding, config)

if __name__ == '__main__':
    main()
