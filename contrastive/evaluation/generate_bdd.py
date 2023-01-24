import pandas as pd
from datetime import datetime

from contrastive.utils.models_database import *


dataset = 'cingulate_ACCpatterns_1'

## construct the database
# folders to look for the models in
# folders = ["/neurospin/dico/agaudin/Runs/04_pointnet/Output", "/neurospin/dico/agaudin/Runs/03_monkeys/Output/analysis_folders/convnet",
# "/neurospin/dico/agaudin/Runs/03_monkeys/Output/analysis_folders/densenet2", "/neurospin/dico/agaudin/Runs/03_monkeys/Output/convnet_exploration"]
folders = ["/neurospin/dico/data/deep_folding/papers/ipmi2023/models/contrastive/trained_on_HCP_half_2/training-different-n",
           "/neurospin/dico/data/deep_folding/papers/ipmi2023/models/contrastive/trained_on_ACCpatterns_0/unsupervised"]
# folders = ["/neurospin/dico/data/deep_folding/papers/ipmi2023/models/beta-VAE"]
bdd = []
visited = []

generate_bdd_models(folders, bdd, visited, verbose=False, dataset=dataset)

bdd = pd.DataFrame(bdd)
print("Number of subjects:", bdd.shape[0])

# remove useless columns
# bdd = post_process_bdd_models(bdd, hard_remove=["partition", "numpy_all", "block_config", "patch_size"], git_branch=True)
bdd = post_process_bdd_models(bdd, hard_remove=["partition", "numpy_all"], git_branch=True)

# bdd = post_process_bdd_models(bdd, hard_remove=[], git_branch=True)


# save the database
name = "HCP-half-2-different-n_evaluation-ACCpatterns-1"
save_path = f"/neurospin/dico/data/deep_folding/papers/ipmi2023/models/contrastive/summary/bdd_{name}.csv"
bdd.to_csv(save_path, index=True)


# write the little readme
with open(f"/neurospin/dico/data/deep_folding/papers/ipmi2023/models/contrastive/summary/README_{name}.txt", 'w') as file:
    file.write("Contient les paramètres de tous les modèles d'intérêt (dossiers précisés en-dessous). La base est faite en sorte que \
seuls les paramètres qui changent entre les modèles soient enregistrés.\n")
    file.write("\n")
    file.write(f"Peformances données pour le dataset {dataset}\n")
    file.write("\n")
    file.write("Généré avec contrastive/evaluation/generate_bdd.py le " + datetime.now().strftime('%d/%m/%Y à %H:%M') + '.\n')
    file.write("\n")
    file.write("Dossiers utilisés : [")
    for folder in folders:
        file.write(folder)
        if folder == folders[-1]:
            file.write(']')
        else:
            file.write(',\n')