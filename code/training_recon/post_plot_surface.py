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

    exps = [Exp(expname='debug',timestemp='2019_10_26_09_59_56',epoch=9900,shape_index = 0)]

    for e in exps:
        post_plot_surface(exp=e,gpu=1,resolution=500)

def post_plot_surface(exp,gpu,resolution):

    GPU_INDEX = gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = '{0}'.format(GPU_INDEX)

    base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'../../exps',exp.expname,exp.timestamp)
    output_path = os.path.join(base_path,'post_process')
    utils.mkdir_ifnotexists(output_path)

    conf_path = os.path.join(base_path,'code/confs/runconf.conf')

    # load conf
    conf = ConfigFactory.parse_file(conf_path)

    epochs = [exp.epoch]

    ds = utils.get_class(conf.get_string('train.dataset'))(data_folder=utils.concat_home_dir('datasets/faust/processed/test'),
                                                              number_of_batchs=1,
                                                              shape_index=exp.shape_index)
    dataloader = torch.utils.data.DataLoader(ds,
                                                  batch_size=1,
                                                  shuffle=True,
                                                  num_workers=1)
    pnts = next(iter(dataloader))

    for i in epochs:
        network = torch.load(os.path.join(base_path,'model/network_{0}_{1}.pt'.format(exp.expname,i)))

        plt.plot_manifold(points=pnts.squeeze(),
                         decoder=network.decoder,
                         path=output_path,
                         epoch=i,
                         in_epoch=0,
                         resolution=resolution,
                         mc_value=0,
                         is_uniform_grid=False,
                         verbose=True,
                         save_html = False)
        plt.plot_cuts(network.decoder, output_path, i, False)
if __name__ == '__main__':
    batch_plot()