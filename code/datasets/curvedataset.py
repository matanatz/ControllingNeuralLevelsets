import torch.utils.data as data
import numpy as np
import os

class CurveDataSet(data.Dataset):
    def __init__(self,
                 data_folder,
                 number_of_batchs,
                 shape_index):
        fns = [x for x in sorted(os.listdir(data_folder))]
        fns = fns[shape_index]
        shape = np.load(os.path.join(data_folder, fns))
        self.data = [shape]


    def __getitem__(self, index):

        return self.data[index]

    def __len__(self):
        return len(self.data)