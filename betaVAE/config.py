
class Config:

    def __init__(self):
        self.batch_size = 64
        self.nb_epoch = 500 #300
        self.kl = 2
        self.n = 10
        self.lr = 2e-4
        self.in_shape = (1, 20, 40, 40) # input size with padding
        self.save_dir = f""
        self.data_dir = "datasets/hcp/crops/2mm/CINGULATE/mask/"
        self.subject_dir = "HCP_half_1bis.csv"
        self.acc_subjects_dir = "datasets/ACCpatterns/crops/2mm/CINGULATE/mask"
        self.test_model_dir = f"#1"
