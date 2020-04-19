import utils.general as utils
import os
from datetime import datetime
from pyhocon import ConfigFactory
from utils.plots import plot_manifold
import sys
import torch
import numpy as np
from model.reconstruction.manifold_network import ManifoldNetwork



class TrainRunner():
    def __init__(self,**kwargs):

        if (type(kwargs['conf']) == str):
            self.conf = ConfigFactory.parse_file(kwargs['conf'])
            self.conf_filename = kwargs['conf']
        else:
            self.conf = kwargs['conf']
        self.number_of_batchs = kwargs['number_of_batchs']
        self.nepochs = kwargs['nepochs']
        self.expname = kwargs['expname']
        self.GPU_INDEX = kwargs['gpu_index']
        self.exps_folder_name = kwargs['exps_folder_name']


        utils.mkdir_ifnotexists(os.path.join(os.path.dirname(os.path.abspath(__file__)),'../../',self.exps_folder_name))

        self.expdir = os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)),'../../',self.exps_folder_name), self.expname)
        utils.mkdir_ifnotexists(self.expdir)
        self.timestamp = '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.now())
        utils.mkdir_ifnotexists(os.path.join(self.expdir, self.timestamp))
        log_dir = os.path.join(self.expdir, self.timestamp, 'log')
        self.log_dir = log_dir
        utils.mkdir_ifnotexists(log_dir)

        self.plots_dir = os.path.join(self.expdir, self.timestamp, 'plots')
        utils.mkdir_ifnotexists(self.plots_dir)

        self.model_path = os.path.join(self.expdir, self.timestamp, 'model')
        utils.mkdir_ifnotexists(self.model_path)

        self.log_fout = open(os.path.join(log_dir, 'log_train.txt'), 'w')

        self.datadir = utils.concat_home_dir('datasets/{0}'.format(self.conf.get_string('train.datapath')))
        os.environ["CUDA_VISIBLE_DEVICES"] = '{0}'.format(self.GPU_INDEX)

        # Backup code
        self.code_path = os.path.join(self.expdir, self.timestamp, 'code')
        utils.mkdir_ifnotexists(self.code_path)
        utils.mkdir_ifnotexists(os.path.join(self.code_path, 'training_recon'))
        utils.mkdir_ifnotexists(os.path.join(self.code_path, 'preprocess'))
        utils.mkdir_ifnotexists(os.path.join(self.code_path, 'utils'))
        utils.mkdir_ifnotexists(os.path.join(self.code_path, 'model'))
        utils.mkdir_ifnotexists(os.path.join(self.code_path, 'datasets'))
        utils.mkdir_ifnotexists(os.path.join(self.code_path, 'confs'))
        os.system("""cp -r ./training_recon/* "{0}" """.format(os.path.join(self.code_path, 'training_recon')))
        os.system("""cp -r ./model/* "{0}" """.format(os.path.join(self.code_path, 'model')))
        os.system("""cp -r ./preprocess/* "{0}" """.format(os.path.join(self.code_path, 'preprocess')))
        os.system("""cp -r ./utils/* "{0}" """.format(os.path.join(self.code_path, 'utils')))
        os.system("""cp -r ./datasets/* "{0}" """.format(os.path.join(self.code_path, 'datasets')))
        os.system("""cp -r ./confs/* "{0}" """.format(os.path.join(self.code_path, 'confs')))
        os.system("""cp -r {0} "{1}" """.format(kwargs['conf'], os.path.join(self.code_path, 'confs/runconf.conf')))

        self.log_string('shell command : {0}'.format(' '.join(sys.argv)))

        ds = utils.get_class(self.conf.get_string('train.dataset'))(data_folder=self.datadir,
                                                                          number_of_batchs=self.number_of_batchs,
                                                                          shape_index=kwargs["shape_index"])
        self.ds_len = len(ds)

        self.dataloader = torch.utils.data.DataLoader(ds,
                                                      batch_size=1,
                                                      shuffle=True,
                                                      num_workers=1)

        self.network = ManifoldNetwork(self.conf.get_config('network'))
        self.network = utils.get_cuda_ifavailable(self.network)
        self.loss = utils.get_class(self.conf.get_string('network.loss.loss_type'))(**self.conf.get_config('network.loss.properties'))

    def run(self):
        print ("runnnig")

        optimizer = torch.optim.Adam(self.network.parameters(),lr=1.0e-4)

        for epoch in range(1,self.nepochs + 1):


            if (epoch) % 100 == 0 and epoch > 1:
                torch.save(self.network, os.path.join(self.model_path, 'network_{0}_{1}.pt'.format(self.expname, epoch)))

                pnts = next(iter(self.dataloader))
                plot_manifold(points=pnts[0],
                             decoder=self.network.decoder,
                             path=self.plots_dir,
                             epoch=epoch,
                             in_epoch=0,
                             **self.conf.get_config('plot'))


            for data_index,data in enumerate(self.dataloader):
                points = data
                points = utils.get_cuda_ifavailable(points)[0]

                outputs = self.network(points)
                loss = self.loss(zerolevelset_points = outputs['zerolevelset_sample_layer'],
                                 genlevelset_points = outputs['genlevelset_sample_layer'],
                                 pc_input = points,
                                 zerolevelset_eval = outputs['zerolevelset_proj_result']['network_eval_on_levelset_points'],
                                 gen_points_eval = outputs['genlevelset_proj_result']['network_eval_on_levelset_points'],
                                 manifold_pnts_pred = outputs['manifold_pnts_pred'],
                                 loss_lambda = None)

                optimizer.zero_grad()
                loss['loss'].backward()
                optimizer.step()
                #scheduler.step()
                print ("expname : {0}".format(self.expname))
                print ("timestamp: {0} , "
                       "epoch : {1}, data_index : {2} , "
                       "loss : {3}, first_part : {4} , "
                       "second_part : {5} ".format(self.timestamp,
                                                   epoch,
                                                   data_index,
                                                   loss['loss'].item(),
                                                   loss['first_term'].item(),
                                                   loss['second_term'].item()))


    def log_string(self,out_str):
        self.log_fout.write(out_str + '\n')
        self.log_fout.flush()
        print(out_str)


