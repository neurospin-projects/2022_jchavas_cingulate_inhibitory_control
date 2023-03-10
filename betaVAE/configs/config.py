
class Config:

    def __init__(self):
        self.batch_size = 64
        self.nb_epoch = 2 #300
        self.kl = 2
        self.n = 4
        self.lr = 2e-4
        self.in_shape = (1, 20, 40, 40) # input size with padding
        self.data_dir = "/path/to/data/directory"
        self.subject_dir = "/path/to/list_of_subjects"
        self.save_dir = "/path/to/saving/directory"

