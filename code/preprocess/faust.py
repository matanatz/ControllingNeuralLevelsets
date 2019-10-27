
import numpy as np
import os
import utils.general as utils
from plyfile import PlyData


homedir = os.environ['HOME']
root = '{0}/data/datasets/faust/MPI-FAUST/training/'.format(homedir)
output_path = '{0}/data/datasets/faust/processed'.format(homedir)
utils.mkdir_ifnotexists(output_path)

output_path_train = os.path.join(output_path,'train')
utils.mkdir_ifnotexists(output_path_train)

output_path_test = os.path.join(output_path,'test')
utils.mkdir_ifnotexists(output_path_test)

scans_path = os.path.join(root,'scans')
file_list = sorted(os.listdir(scans_path))
for file in file_list:

    if (file.split('.')[-1] == 'ply'):
        print (file)
        plydata = PlyData.read(os.path.join(scans_path,file))

        point_set = np.array([[x[0],x[1],x[2]] for x in  plydata.elements[0].data])
        idx = np.random.choice(np.arange(point_set.shape[0]),100000,False)
        other_idx = np.array(list(set(range(point_set.shape[0])).difference(set(idx.tolist()))))
        point_set_train = point_set[idx]
        point_set_train = point_set_train - np.mean(point_set_train,axis=0,keepdims=True)
        np.save(os.path.join(output_path_train,file),point_set_train)
        point_set_test = point_set[other_idx]
        point_set_test = point_set_test - np.mean(point_set_test, axis=0, keepdims=True)
        np.save(os.path.join(output_path_test, file), point_set_test)

print ("finished")