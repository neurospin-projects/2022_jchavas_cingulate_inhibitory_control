import os
import json
import yaml

from tensorflow.python.summary.summary_iterator import summary_iterator

# functions to create a database containing all the models
# These functions are used in the generate_bdd notebook


def get_subdirs(directory):
    sub_dirs = os.listdir(directory)
    sub_dirs = [os.path.join(directory, name) for name in sub_dirs]
    sub_dirs = [path for path in sub_dirs if os.path.isdir(path)] # remove files
    return sub_dirs


def get_path2logs(model_path):
    # get the right templating for the log files
    if os.path.exists(model_path + "/logs/default/version_0"):
        path = model_path + "/logs/default/version_0"
    elif os.path.exists(model_path + "/logs/lightning_logs/version_0"):
        path = model_path + "/logs/lightning_logs/version_0"
    else:
        raise ValueError("No logs at this address OR different templating for the save path.")
    return path


def get_loss(model_path, save=False, verbose=False):

    path = get_path2logs(model_path)
    
    # get the file
    for file in os.listdir(path):
        if 'events.out' in file:
            full_path = os.path.join(path, file)
            if verbose:
                print("Treating", model_path)
    
    if full_path == None:
        print(f"No corresponding logs for the model at {model_path}")
    
    loss_train = 0
    loss_val = 0

    for e in summary_iterator(full_path):
        for v in e.summary.value:
            if v.tag == 'Loss/Validation':
                loss_val = v.simple_value
            elif v.tag == 'Loss/Train':
                loss_train = v.simple_value
    
    if save:
        final_losses = {"train_loss": loss_train,
                        "val_loss": loss_val}
        with open(path+"/final_losses.json", 'w') as file:
            json.dump(final_losses, file)
        if verbose:
            print(final_losses)
    else:
        return loss_train, loss_val


def process_model(model_path, dataset='cingulate_ACCpatterns', verbose=True):
    # generate a dictionnary with the model's parameters and performances
    model_dict = {}
    model_dict['model_path'] = model_path

    # read performances
    with open(model_path + f"/{dataset}_embeddings/values.json", 'r') as file:
        values = json.load(file)
        decomposed_values = {'auc': values['cross_val_auc'][0],
                            'auc_std': values['cross_val_auc'][1],
                            'accuracy': values['cross_val_total_accuracy'][0],
                            'accuracy_std': values['cross_val_total_accuracy'][1]}
        model_dict.update(decomposed_values)
    
    # read parameters
    with open(model_path+'/partial_config.yaml', 'r') as file2:
        partial_config = yaml.load(file2, Loader=yaml.FullLoader)
        model_dict.update(partial_config)
    
    # compute losses if necessary
    log_path = get_path2logs(model_path)
    if not os.path.exists(os.path.join(log_path, "final_losses.json")):
        if verbose:
            print(f"Get the losses for {model_path}.")
        get_loss(model_path, save=True, verbose=verbose)
    
    # get the final losses
    with open(os.path.join(log_path, "final_losses.json"), 'r') as file3:
        losses = json.load(file3)
        model_dict.update(losses)

    return model_dict


def generate_bdd_models(folders, bdd_models, visited, dataset='cingulate_ACCpatterns', verbose=True):
    # fill the dictionnary bdd_models with the parameters and performances of all the bdd models
    # depth first exploration of folders to treat all the models in it
    
    if verbose:
        print("Start", len(folders), len(bdd_models))

    while folders != []:
        # remove folders already treated
        folders = [folder for folder in folders if folder not in visited]
        
        # condition as folders can be emptied by the previous line
        if folders != []:
            dir_path = folders.pop()
            visited.append(dir_path)
            
            # checks if directory
            if os.path.isdir(dir_path):
                # check if directory associated to a model
                if os.path.exists(dir_path+'/.hydra/config.yaml'):
                    print("Treating", dir_path)
                    # check if values and parameters computed for the model
                    if os.path.exists(dir_path + f"/{dataset}_embeddings/values.json"):
                        model_dict = process_model(dir_path)
                        bdd_models.append(model_dict)
                        if verbose:
                            print("End model", len(folders), len(bdd_models))

                    else:
                        print(f"Model does not have embeddings and their evaluation OR \
they are done with another database than {dataset}")

                else:
                    print(f"{dir_path} not associated to a model. Continue")
                    new_dirs = get_subdirs(dir_path)
                    folders.extend(new_dirs)
                    # remove folders already treated
                    folders = [folder for folder in folders if folder not in visited]
                    if verbose:
                        print("End recursive", len(folders), len(bdd_models))
                    
                    generate_bdd_models(folders, bdd_models, visited, dataset=dataset, verbose=verbose)
            
            else:
                print(f"{dir_path} is a file. Continue.")
                if verbose:
                    print("End file", len(bdd_models))


def post_process_bdd_models(bdd_models, hard_remove=[], git_branch=False):
    # specify dataset if not done
    if "dataset_name" in bdd_models.columns:
        bdd_models.numpy_all.fillna(value="osef", inplace=True)
        bdd_models.dataset_name.fillna(value="cingulate_HCP_half_1", inplace=True)
        bdd_models.loc[bdd_models.numpy_all.str.contains('1mm'), 'dataset_name'] = "cingulate_HCP_1mm"
    
    # hard_remove contains columns you want to remove by hand
    bdd_models = bdd_models.drop(columns=hard_remove)

    # remove duplicates (normally not needed)
    bdd_models.drop_duplicates(inplace=True, ignore_index=True)

    # deal with '[' and ']'
    # TODO

    # specify git branch
    if git_branch:
        bdd_models['git_branch'] = ['Run_03_aymeric' for i in range(bdd_models.shape[0])]
        bdd_models.loc[bdd_models.backbone_name.isna(), 'git_branch'] = 'Run_43_joel'
        bdd_models.loc[bdd_models.backbone_name == 'pointnet', 'git_branch'] = 'pointnet'


    # remove columns where the values never change
    remove = []
    for col in bdd_models.columns:
        col_values = bdd_models[col].dropna().unique()
        if len(col_values) <= 1:
            remove.append(col)
    bdd_models = bdd_models.drop(columns=remove)

    # sort by model_path
    bdd_models.sort_values(by="model_path", axis=0, inplace=True, ignore_index=True)


    return bdd_models