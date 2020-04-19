import sys
sys.path.append('../code')
import argparse
import torch
import utils.general as utils
import utils.plots as plt
from pyhocon import ConfigFactory
import os

class Exp:
    def __init__(self,expname,timestemp,epoch,shape_index):
        self.expname = expname
        self.timestamp = timestemp
        self.epoch = epoch
        self.shape_index = shape_index

def batch_plot():

    exps = [Exp(expname='lambda3_shape6_withzero',timestemp='2020_04_18_09_43_51',epoch=5000,shape_index = 6)]

    for e in exps:
        post_plot_surface(exp=e,gpu=4,resolution=500)

def post_plot_surface(expname,epoch,gpu,resolution,base_path,shape_index):


    GPU_INDEX = gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = '{0}'.format(GPU_INDEX)


    output_path = os.path.join(base_path,'post_process')
    utils.mkdir_ifnotexists(output_path)

    conf_path = os.path.join(base_path,'code/confs/runconf.conf')

    # load conf
    conf = ConfigFactory.parse_file(conf_path)

    epochs = [epoch]

    ds = utils.get_class(conf.get_string('train.dataset'))(data_folder=utils.concat_home_dir('datasets/faust/processed/test'),
                                                              number_of_batchs=1,
                                                              shape_index=shape_index)
    dataloader = torch.utils.data.DataLoader(ds,
                                                  batch_size=1,
                                                  shuffle=True,
                                                  num_workers=1)
    pnts = next(iter(dataloader))

    for i in epochs:
        network = torch.load(os.path.join(base_path,'model/network_{0}_{1}.pt'.format(expname,i)))

        plt.plot_manifold(points=pnts.squeeze(),
                         decoder=network.decoder,
                         path=output_path,
                         epoch=i,
                         in_epoch=0,
                         resolution=resolution,
                         mc_value=0,
                         is_uniform_grid=True,
                         verbose=True,
                         save_html = False)
        plt.plot_cuts(network.decoder, output_path, i, False)
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--expname', type=str, default="recon", help='experiement name')
    parser.add_argument('--epoch', type=int, default=5500)
    parser.add_argument('--shape_index', type=int, default=0)
    parser.add_argument('--timestamp', type=str, default='latest')
    parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
    parser.add_argument('--res', type=int, default=500, help='GPU to use [default: GPU 0]')

    opt = parser.parse_args()
    base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../exps', opt.expname)
    if opt.timestamp == 'latest':
        timestamps = os.listdir(base_path)
        timestamp = sorted(timestamps)[-1]
    else:
        timestamp = opt.timestamp

    base_path = os.path.join(base_path,timestamp)
    post_plot_surface(opt.expname, opt.epoch, opt.gpu, opt.res, base_path,opt.shape_index)
