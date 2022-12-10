
class Config:

    def __init__(self):
        self.batch_size = 64
        self.nb_epoch = 500 #300
        self.kl = 2
        self.n = 10
        self.lr = 2e-4
        self.in_shape = (1, 20, 40, 40) # input size with padding
        self.save_dir = f"/path/to/save/dir"
        self.data_dir = "/path/to/data/dir"
        self.subject_dir = "path/to/subject/csv"
        self.acc_subjects_dir = "path/to/acc/mask"
        self.test_model_dir = f"/path/to/model"
