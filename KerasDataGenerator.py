import keras
import numpy as np

class KerasDataGenerator(keras.utils.Sequence):
    def __init__(self, x, ydet, ytype, yclass, batch_size):
        self.x = x
        self.ydet = ydet
        self.ytype = ytype
        self.yclass = yclass
        self.batch_size = batch_size
    
    def __len__(self):
        return (np.ceil(len(self.x)/self.batch_size)).astype(np.int)

    def __getitem__(self, idx):
        return self.x[idx * self.batch_size : (idx + 1) * self.batch_size], [self.ydet[idx * self.batch_size : (idx + 1) * self.batch_size], self.ytype[idx * self.batch_size : (idx + 1) * self.batch_size], self.yclass[idx * self.batch_size : (idx + 1) * self.batch_size]]