import argparse
import sys
sys.path.append('../code')
from training_recon.train import TrainRunner



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--number_of_batchs', type=int, default=10, help='input batch size')
    parser.add_argument('--nepoch', type=int, default=10000, help='number of epochs to train for')
    parser.add_argument('--conf', type=str, default='./confs/recon/default.conf')
    parser.add_argument('--expname', type=str, default='debug')
    parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
    parser.add_argument('--shape_index', type=int, default=0, help='GPU to use [default: GPU 0]')


    opt = parser.parse_args()
    # TODO save parser argumernts
    trainrunner = TrainRunner(conf=opt.conf,
                              number_of_batchs=opt.number_of_batchs,
                            nepochs=opt.nepoch,
                            expname=opt.expname,
                            gpu_index=opt.gpu,
                            exps_folder_name='exps',
                            shape_index=opt.shape_index)


    trainrunner.run()