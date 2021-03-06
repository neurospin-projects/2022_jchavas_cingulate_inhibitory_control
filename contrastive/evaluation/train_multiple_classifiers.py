import re
import hydra
import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import json
import os

from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split

from contrastive.models.binary_classifier import BinaryClassifier 

from contrastive.data.utils import read_labels

from contrastive.utils.config import process_config
from contrastive.utils.logs import set_root_logger_level



def load_and_format_embeddings(dir_path, labels_path, config):
    # if targeting directly the target csv file
    if not os.path.isdir(dir_path):
        embeddings = pd.read_csv(dir_path, index_col=0)
    # if only giving the directory (implies constraints on the file name)
    else:
        if os.path.exists(dir_path+'/full_embeddings.csv'):
            embeddings = pd.read_csv(dir_path+'/full_embeddings.csv', index_col=0)
        elif os.path.exists(dir_path+'/pca_embeddings.csv'):
            embeddings = pd.read_csv(dir_path+'/pca_embeddings.csv', index_col=0)
        else:
            train_embeddings = pd.read_csv(dir_path+'/train_embeddings.csv', index_col=0)
            val_embeddings = pd.read_csv(dir_path+'/val_embeddings.csv', index_col=0)
            test_embeddings = pd.read_csv(dir_path+'/test_embeddings.csv', index_col=0)

            # regroup them in one dataframe (discuss with Joël)
            embeddings = pd.concat([train_embeddings, val_embeddings, test_embeddings],
                                axis=0, ignore_index=False)
    
    names_col = 'ID' if 'ID' in embeddings.columns else 'Subject'
    embeddings.sort_index(inplace=True)
    print("sorted embeddings:", embeddings.head())

    # get the labels (0 = no paracingulate, 1 = paracingulate)
    # /!\ use read_labels
    labels = read_labels(labels_path, config.subject_column_name, config.label_names)
    labels = pd.DataFrame(labels.values, columns=['Subject', 'label'])
    labels.sort_values(by='Subject', inplace=True, ignore_index=True)
    print("sorted labels", labels.head())
    # /!\ multiple labels is not handled

    # create train-test datasets
    if config.classifier_test_size:
        embeddings_train, embeddings_test, labels_train, labels_test = \
            train_test_split(embeddings, labels, test_size=config.classifier_test_size, 
            random_state=config.classifier_seed)
    else: # no train-test sets for the classifier
        embeddings_train = embeddings_test = embeddings
        labels_train = labels_test = labels

    # cast the dataset to the torch format
    X_train =  torch.from_numpy(embeddings_train.loc[:, embeddings_train.columns != names_col].values).type(torch.FloatTensor)
    X_test =  torch.from_numpy(embeddings_test.loc[:, embeddings_test.columns != names_col].values).type(torch.FloatTensor)
    Y_train = torch.from_numpy(labels_train.label.values.astype('float32')).type(torch.FloatTensor)
    Y_test = torch.from_numpy(labels_test.label.values.astype('float32')).type(torch.FloatTensor)

    return X_train, X_test, Y_train, Y_test, labels_train, labels_test



def compute_indicators(Y, labels_pred):
    # compute ROC curve and auc
    labels_true = Y.detach_().numpy()
    curves = roc_curve(labels_true, labels_pred)
    roc_auc = roc_auc_score(labels_true, labels_pred)

    # choose labels predicted with frontier = 0.5
    labels_pred = (labels_pred >= 0.5).astype('int')
    # compute accuracy
    accuracy = accuracy_score(labels_true, labels_pred)
    return curves, roc_auc, accuracy



def compute_auc(column, label_col=None):
    print("COMPUTE AUC")
    print(label_col.head())
    print(column.head())
    return roc_auc_score(label_col, column)

# get a model with performance that is representative of the group
def get_average_model(labels_df):
    aucs = labels_df.apply(compute_auc, args=[labels_df.label])
    aucs = aucs[aucs.index != 'label']
    aucs = aucs[aucs == aucs.quantile(interpolation='nearest')]
    return(aucs.index[0])



# would highly benefit from paralellisation
@hydra.main(config_name='config_no_save', config_path="../configs")
def train_classifiers(config):
    config = process_config(config)

    set_root_logger_level(config.verbose)

    # set up load and save paths
    train_embs_path = config.training_embeddings
    train_lab_paths = config.training_labels
    # if not specified, the embeddings the results are created from are the ones used for training

    EoI_path = config.embeddings_of_interest if config.embeddings_of_interest else train_embs_path
    LoI_path = config.labels_of_interest if config.labels_of_interest else train_lab_paths

    # if not specified, the outputs of the classifier will be stored next to the embeddings
    # used to generate them
    results_save_path = config.results_save_path if config.results_save_path else EoI_path
    if not os.path.isdir(results_save_path):
        results_save_path = os.path.dirname(results_save_path)
    
    
    # import the embeddings (supposed to be already computed)
    X_train, X_test, Y_train, Y_test, labels_train, labels_test = \
        load_and_format_embeddings(train_embs_path, train_lab_paths, config)


    # create objects that will be filled during the loop
    train_prediction_matrix = np.zeros((labels_train.shape[0], config.n_repeat))
    test_prediction_matrix = np.zeros((labels_test.shape[0], config.n_repeat))

    Curves = {'train': [],
              'test': []}
    aucs = {'train': [],
            'test': []}
    accuracies = {'train': [],
                  'test': []}


    for i in range(config.n_repeat):
        print("model number", i)
        # create and train the model
        if i == 0:
            hidden_layers = list(config.classifier_hidden_layers)
            layers_shapes = [np.shape(X_train)[1]]+hidden_layers+[1]
        
        bin_class = BinaryClassifier(layers_shapes,
                                    activation=config.classifier_activation,
                                    loss=config.classifier_loss)

        if i == 0:
            print("model", bin_class)

        class_train_set = TensorDataset(X_train, Y_train)
        train_loader = DataLoader(class_train_set, batch_size=config.class_batch_size)

        trainer = pl.Trainer(max_epochs=config.class_max_epochs, logger=False, enable_checkpointing=False)
        trainer.fit(model=bin_class, train_dataloaders=train_loader)


        # load new embeddings and labels if needed
        if (EoI_path == train_embs_path) and (LoI_path == train_lab_paths):
            pass
        else:
            pass
            """# /!\ DOESN'T WORK !!!!
            # load embeddings of interest
            X,Y,n_train,n_val,_ = load_and_format_embeddings(EoI_path, LoI_path, config)
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1.0, random_state=24)"""

        # predict labels with the classifier (both for train and test sets)
        labels_pred_train = bin_class.forward(X_train).detach().numpy()
        labels_pred_test = bin_class.forward(X_test).detach().numpy()
        # save the predicted labels
        train_prediction_matrix[:,i] = labels_pred_train.flatten()
        test_prediction_matrix[:,i] = labels_pred_test.flatten()

        # compute indicators for train
        curves, roc_auc, accuracy = compute_indicators(Y_train, labels_pred_train)
        Curves['train'].append(curves)
        aucs['train'].append(roc_auc)
        accuracies['train'].append(accuracy)

        # compute indicators for test
        curves, roc_auc, accuracy = compute_indicators(Y_test, labels_pred_test)
        Curves['test'].append(curves)
        aucs['test'].append(roc_auc)
        accuracies['test'].append(accuracy)


        # plot the histogram of predicted values
        """with_paracingulate = labels[labels.label == 1]
        without_paracingulate = labels[labels.label == 0]

        print(with_paracingulate.shape[0], "-", without_paracingulate.shape[0])

        x_min = min(0, np.min(labels.predicted))
        x_max = max(1, np.max(labels.predicted))

        plt.figure()
        plt.hist(without_paracingulate.predicted, bins=np.arange(x_min,x_max,0.01), alpha=0.6)
        plt.hist(with_paracingulate.predicted, bins=np.arange(x_min,x_max,0.01), alpha=0.6, color='r')
        plt.legend(['without_paracingulate', "with_paracingulate"])

        ax = plt.gca()
        plt.vlines([0.5], ax.get_ylim()[0], ax.get_ylim()[1], color='black')

        plt.savefig(results_save_path+"/prediction_histogram.png")"""

    # add the predictions to the df where the true values are
    columns_names = ["predicted_"+str(i) for i in range(config.n_repeat)]
    train_preds = pd.DataFrame(train_prediction_matrix, columns=columns_names, index=labels_train.index)
    labels_train = pd.concat([labels_train, train_preds], axis=1)

    test_preds = pd.DataFrame(test_prediction_matrix, columns=columns_names, index=labels_test.index)
    labels_test = pd.concat([labels_test, test_preds], axis=1)


    # evaluation of the aggregation of the models
    values = {}

    for mode in ['train', 'test']:
        if mode == 'train':
            labels = labels_train
        elif mode == 'test':
            labels = labels_test
        
        labels_true = labels.label.values.astype('float64')
        
        # compute agregated models
        predicted_labels = labels[columns_names]

        labels['median_pred'] = predicted_labels.median(axis=1)
        labels['mean_pred'] = predicted_labels.mean(axis=1)

        # plot ROC curves
        plt.figure()

        # ROC curves of all models
        for curves in Curves[mode]:
            plt.plot(curves[0], curves[1], color='grey', alpha=0.1)
        plt.plot([0,1],[0,1],color='r', linestyle='dashed')

        # get the average model (with AUC as a criteria)
        # /!\ This model is a classifier that exists in the pool != 'mean_pred' and 'median_pred'
        average_model = get_average_model(labels[['label'] + columns_names].astype('float64'))
        roc_curve_average = roc_curve(labels_true, labels[average_model].values)
        # ROC curves of "special" models
        roc_curve_median = roc_curve(labels_true, labels.median_pred.values)
        roc_curve_mean = roc_curve(labels_true, labels.mean_pred.values)
        
        plt.plot(roc_curve_average[0], roc_curve_average[1], color='red', alpha=0.5, label='average model')
        plt.plot(roc_curve_median[0], roc_curve_median[1], color='blue', label='agregated model (median)')
        plt.plot(roc_curve_mean[0], roc_curve_mean[1], color='black', label='agregated model (mean)')
        plt.legend()
        plt.title(f"{mode} ROC curves")
        plt.savefig(results_save_path+f"/{mode}_ROC_curves.png")

        # compute accuracy and area under the curve
        print(f"{mode} accuracy", np.mean(accuracies[mode]), np.std(accuracies[mode]))
        print(f"{mode} AUC", np.mean(aucs[mode]), np.std(aucs[mode]))

        values[f'{mode}_total_accuracy'] = [np.mean(accuracies[mode]), np.std(accuracies[mode])]
        values[f'{mode}_auc'] = [np.mean(aucs[mode]), np.std(aucs[mode])]

        # save predicted labels
        labels.to_csv(results_save_path+f"/{mode}_predicted_labels.csv", index=False)

    with open(results_save_path+"/values.json", 'w+') as file:
        json.dump(values, file)

    plt.show()
    plt.close('all')



if __name__ == "__main__":
    train_classifiers()