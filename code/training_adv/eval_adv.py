import utils.general as utils
import copy
import os
from pyhocon import ConfigFactory
import sys
import torch
import logging
import torchvision
import pandas as pd
import numpy as np

class EvalRunner():
    def __init__(self,**kwargs):

        if (type(kwargs['conf']) == str):
            self.conf = ConfigFactory.parse_file(kwargs['conf'])
            self.conf_filename = kwargs['conf']
        else:
            self.conf = kwargs['conf']

        self.expname = kwargs['expname']
        self.GPU_INDEX = kwargs['gpu_index']
        self.batch_size = kwargs['batch_size']
        self.exps_folder_name = kwargs['exps_folder_name']

        self.expdir = os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)),'../../',self.exps_folder_name), self.expname)

        log_dir = os.path.join(self.expdir, 'log')
        self.log_dir = log_dir
        utils.mkdir_ifnotexists(log_dir)

        self.learning_test_log = os.path.join(self.log_dir,'learning_test_log.csv')
        self.learning_test_adv_log = os.path.join(self.log_dir, 'learning_test_adv_log.csv')

        self.datadir = utils.concat_home_dir('{0}'.format(self.conf.get_string('train.datapath')))
        os.environ["CUDA_VISIBLE_DEVICES"] = '{0}'.format(self.GPU_INDEX)

        logging.info('shell command : {0}'.format(' '.join(sys.argv)))

        ds_test = utils.get_class(self.conf.get_string('train.dataset'))(self.datadir,
                                                                                train=False,
                                                                                download=True,
                                                                                transform=torchvision.transforms.ToTensor())

        self.test_dataloader = torch.utils.data.DataLoader(ds_test,
                                                      batch_size=self.batch_size,
                                                      shuffle=False,
                                                      num_workers=kwargs['threads'],
                                                      drop_last=False)

        self.network = utils.get_class(self.conf.get_string('network.type'))(self.conf.get_config('network'))
        self.network = utils.get_cuda_ifavailable(self.network)

        saved_model_state = torch.load(
            os.path.join(self.expdir, str(kwargs['checkpoint']) + ".pth"))
        self.network.load_state_dict(saved_model_state["model_state_dict"])
        self.startepoch = saved_model_state["epoch"]

        # blackbox
        if (self.conf.get_bool('adversarial.blackbox.enabled')):
            self.blackbox_networks = []
            for network_name in self.conf.get_config('adversarial.blackbox.networks').keys():
                blackbox_conf = self.conf.get_config('adversarial.blackbox.networks').get_config(network_name)
                blackbox_network = utils.get_cuda_ifavailable(utils.get_class(self.conf.get_string('adversarial.blackbox.type'))(blackbox_conf))

                filename = '../evaluation/{0}/{1}.pth'.format(
                    blackbox_conf.get_string("expname"),
                    blackbox_conf.get_string("epoch"))
                if (torch.cuda.is_available()):
                    data = torch.load(filename)
                else:
                    data = torch.load(filename,map_location=torch.device('cpu'))

                model_state_black = {k.replace('classifier', 'model').replace('feature_extractor', 'model'): v for k, v in
                               data["model_state_dict"].items()}
                blackbox_network.load_state_dict(model_state_black)
                self.blackbox_networks.append(blackbox_network)

    def eval_step(self,data):
        input = utils.get_cuda_ifavailable(data[0])
        target = utils.get_cuda_ifavailable(data[1])

        clonemodel = copy.deepcopy(self.network.model)
        network_output = clonemodel(input)

        correct = self.network.get_correct(network_output,target).sum().item()
        seen_examples = target.shape[0]
        return utils.dict_to_nametuple("EvalStep",
                                       dict(correct=correct, seen_examples=seen_examples))

    def adv_step(self, model_adv, data, attack,**kwargs):
        input = utils.get_cuda_ifavailable(data[0])
        target = utils.get_cuda_ifavailable(data[1])
        kwargs['loss_fn'] = utils.get_class(kwargs['loss_fn'])(batch_size=self.batch_size, **kwargs['loss_fn_prop'])
        kwargs.pop('loss_fn_prop', None)
        restarts = kwargs['restarts'] if 'restarts' in kwargs.keys() else 1
        kwargs.pop('restarts', None)

        model = copy.deepcopy(self.network.model)
        model_adv = copy.deepcopy(model_adv)
        model_adv.zero_grad()
        adversary = utils.get_class(attack)(model_adv, **kwargs)

        correct_ind = torch.ones_like(target).byte().cuda()
        for _ in range(restarts):
            adv_samples = adversary.perturb(input, target)
            network_output = model(adv_samples)
            a = self.network.get_correct(network_output,target=target).byte()
            correct_ind &= a
        correct = correct_ind.sum().item()

        seen_examples = target.shape[0]

        return utils.dict_to_nametuple("AdvStep",
                                       dict(correct=correct, seen_examples=seen_examples))

    def run(self):
        repeats = 10

        test_accuracy_log = []
        adv_conf = self.conf.get_config('adversarial.attacks')
        adv_keys = []
        for attack_name in adv_conf.keys():
            attack_conf = adv_conf.get_config(attack_name)
            if not attack_conf.get_bool("is_whitebox") and self.conf.get_bool('adversarial.blackbox.enabled'):
                for network_name in self.conf.get_config('adversarial.blackbox.networks').keys():
                    adv_keys.append('{0}_{1}'.format(attack_name, network_name))
            else:
                adv_keys.append(attack_name)

        test_adv_accuracy_log = {k + '_accuracy':[] for k in adv_keys}

        epoch = self.startepoch

        logging.info("epoch {}...".format(epoch))

        self.network.eval()

        for r in range(repeats):
            print("---- repeat {} ----".format(r))
            correct = 0
            seen_examples = 0

            attack_res = {k: dict(correct=0) for k in adv_keys}

            for data_index, data in enumerate(self.test_dataloader):
                eval_res = self.eval_step(data)
                correct += eval_res.correct
                seen_examples += eval_res.seen_examples

                for attack_name in adv_conf.keys():
                    attack_conf = adv_conf.get_config(attack_name)
                    if attack_conf.get_bool("is_whitebox"):
                        model_adv = self.network.model
                        attack_res[attack_name]['correct'] += self.adv_step(model_adv, data,attack_conf.get_string('attack'),
                                                                            **attack_conf.get_config('props')).correct
                    elif self.conf.get_bool('adversarial.blackbox.enabled'):
                        for idx, network_name in enumerate(self.conf.get_config('adversarial.blackbox.networks').keys()):
                            model_adv = self.blackbox_networks[idx].model
                            attack_res['{0}_{1}'.format(attack_name, network_name)]['correct'] += self.adv_step(model_adv, data,
                                                                                attack_conf.get_string('attack'),
                                                                                **attack_conf.get_config(
                                                                                    'props')).correct

            for attack_name in adv_keys:
                test_adv_accuracy_log[attack_name + "_accuracy"].append(float(attack_res[attack_name]['correct'])/seen_examples)
            test_accuracy_log.append(float(correct)/seen_examples)

        test_accuracy_array = np.array(test_accuracy_log)
        print('test_accuracy: mean {0:.4f} std {0:.6f}'.format(test_accuracy_array.mean(), test_accuracy_array.std()))
        pd.DataFrame(test_accuracy_log).to_csv(self.learning_test_log)

        for l in test_adv_accuracy_log:
            test_adv_array = np.array(test_adv_accuracy_log[l])
            print('{0}: mean {1:.4f} std {2:.6f}'.format(l, test_adv_array.mean(), test_adv_array.std()))
        pd.DataFrame(test_adv_accuracy_log).to_csv(self.learning_test_adv_log)