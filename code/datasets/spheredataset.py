import torch
import torch.utils.data as data
import numpy as np
import os
from utils.general import *
from utils.plots import plot_threed_scatter

class SphereDataSet(data.Dataset):
    def __init__(self,
                 data_folder = None,
                 batch_size = 1,
                 shape_index =0 ,
                 is_train = True):


        self.is_train = is_train
        self.data_folder = data_folder

        np.random.seed(1)

        z = np.random.uniform(-0.5, 0.5, [30, 1])
        phi = np.random.uniform(0, 2 * np.pi, [30, 1])
        x = np.sqrt(0.5 ** 2 - z ** 2) * np.cos(phi)
        y = np.sqrt(0.5 ** 2 - z ** 2) * np.sin(phi)
        points = np.concatenate([x, y, z], axis=-1)
        self.points = points.astype(np.float32)





    def __getitem__(self, index):
        point_set = torch.from_numpy(self.points)

        return point_set


    def __len__(self):
        return 1
if __name__ == '__main__':
    ds = SphereDataSet()

    dataloader = torch.utils.data.DataLoader(ds, batch_size=1,
                                             shuffle=True, num_workers=1)
    print (len(dataloader))
    for data in dataloader:
        plot_threed_scatter(data.squeeze(),'.',0,0)
        print (data.shape)