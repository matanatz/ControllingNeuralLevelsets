import torch
import torch.utils.data as data
import numpy as np
import os
from utils.general import *
from utils.plots import plot_threed_scatter

class FaustDataSet(data.Dataset):
    def __init__(self,
                 data_folder,
                 number_of_batchs,
                 shape_index):

        self.data_folder = data_folder
        self.datapath = []

        fns = [x for x in sorted(os.listdir(data_folder))]
        fns = fns[shape_index]
        shape = np.load(os.path.join(data_folder, fns))

        idx = np.arange(shape.shape[0])

        shape = shape[idx,:]

        batchs = []
        for i in range(number_of_batchs):
            batchs.append(shape[i*(shape.shape[0]//number_of_batchs):(i+1)*(shape.shape[0]//number_of_batchs)])

        self.batchs = batchs

    def __getitem__(self, index):
        point_set = torch.from_numpy(self.batchs[index])

        return point_set


    def __len__(self):
        return len(self.batchs)
if __name__ == '__main__':
    ds = FaustDataSet(data_folder = concat_home_dir('datasets/faust/processed_100k_nofps'),
                      number_of_batchs= 1,
                      shape_index = 0,
                      is_train = True)

    dataloader = torch.utils.data.DataLoader(ds, batch_size=1,
                                             shuffle=True, num_workers=1)
    print (len(dataloader))
    for data in dataloader:
        print (data.shape)
        plot_threed_scatter(data.squeeze(),'.',0,0)