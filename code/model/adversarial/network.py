import utils.general as utils
import torch.nn as nn
import numpy as np
from model.common.sample_network import SampleNetwork
from model.common.loss import MarginLoss
import copy



class MarginWithSampleNetwork(nn.Module):
    def __init__(self,conf):
        super().__init__()
        self.sample_network = SampleNetwork()
        self.conf = conf
        self.model = utils.get_class(self.conf.get_string('model'))()
        self.eps_correct_start = self.conf.get_float('adversarial_training.eps')
        self.eps_correct_end = self.conf.get_float('adversarial_training.eps_correct_fix.eps_end')

        self.eps_correct = np.linspace(start=self.eps_correct_start,stop=self.eps_correct_end,num=self.conf.get_int('adversarial_training.eps_correct_fix.epoch'))

        self.eps_wrong_start = self.conf.get_float('adversarial_training.eps_wrong')
        self.eps_wrong_end = self.conf.get_float('adversarial_training.eps_wrong_fix.eps_end')
        self.eps_wrong = np.linspace(self.eps_wrong_start, self.eps_wrong_end,
                                       self.conf.get_int('adversarial_training.eps_wrong_fix.epoch'))

    def get_correct(self,network_output,target):
        if ('output' in str(type(network_output))):
            correct = network_output.correct
        else:
            pred = network_output.argmax(dim=1)  # get the index of the max log-probability
            correct = pred.eq(target.view_as(pred))
        return correct

    def forward(self, input, **kwargs):
        if (kwargs['epoch'] >= self.conf.get_int('adversarial_training.eps_correct_fix.epoch')):
            eps_correct = self.eps_correct[-1]
        else:
            eps_correct = self.eps_correct[kwargs['epoch']]

        if (kwargs['epoch'] >= self.conf.get_int('adversarial_training.eps_wrong_fix.epoch')):
            eps_wrong = self.eps_wrong[-1]
        else:
            eps_wrong = self.eps_wrong[kwargs['epoch']]

        debug = {}
        debug['eps_correct'] = [eps_correct]
        debug['eps_wrong'] = [eps_wrong]

        margin_func = MarginLoss(reduction='none', correct_negative=True, batch_size=None,in_adv=True)

        clone_model = copy.deepcopy(self.model)

        # Project to the zero levelset of evaluation mode network
        clone_model.eval()
        output = clone_model(input).detach()

        output = margin_func(output, target=kwargs['target'])
        debug['imgs_values_mean'] = [output.mean().item()]
        debug['imgs_values_max'] = [output.max().item()]
        debug['imgs_values_min'] = [output.min().item()]
        correct = (output < 0).detach()

        cat_input = input

        target = kwargs['target']
        clone_model.zero_grad()

        projection_result = utils.get_class(self.conf.get_string('adversarial_training.projection_method'))(1.0e-5,self.conf.get_int('adversarial_training.nb_iter')).project_points(cat_input,
                                                                           clone_model,
                                                                           margin_func,
                                                                           target,
                                                                           self.conf.get_float('adversarial_training.alpha'),
                                                                           eps_correct,
                                                                           eps_wrong,
                                                                           self.conf.get_float('adversarial_training.eps_base'),
                                                                           self.conf.get_string('adversarial_training.init'))

        debug = dict(**debug,**projection_result['debug'])

        if (projection_result['levelset_points'] is None):
            return utils.dict_to_nametuple("output", dict(zero_pnts_sample_network=None,
                                                          effective__samples=None,
                                                          zero_pnts_grad=None,
                                                          zero_pnts_sign=None,
                                                          correct=correct,
                                                          eps=(eps_correct * correct.float() + eps_wrong * (~correct).float())[projection_result['sign_changed']]
                                                          .view(projection_result['sign_changed'].sum().item(), 1, 1,
                                                                1),
                                                          debug=debug))
        else:
            clone_model = copy.deepcopy(self.model)
            pnts = projection_result['levelset_points']
            pnts.requires_grad_(True)
            clone_model.zero_grad()
            eval = margin_func(clone_model(pnts),target[projection_result['sign_changed']])
            eval_sum = eval.sum()
            eval_sum.backward()
            sample_netwrok_params = {}
            sample_netwrok_params['levelset_points_Dx'] = pnts.grad.data.detach()
            sample_netwrok_params['network_eval_on_levelset_points'] = eval.detach()
            sample_netwrok_params['levelset_points'] = pnts.detach()
            sample_netwrok_params['target'] = target[projection_result['sign_changed']]
            zero_pnts_sample_network = self.sample_network(network=self.model,
                                                                  additional_func=MarginLoss(reduction='none',
                                                                                             correct_negative=True,
                                                                                             batch_size=None,in_adv=True),
                                                           **sample_netwrok_params)


            return utils.dict_to_nametuple("output", dict(zero_pnts_sample_network=zero_pnts_sample_network,
                                                          effective_samples=cat_input[projection_result['sign_changed']].detach(),
                                                          zero_pnts_grad = sample_netwrok_params['levelset_points_Dx'].detach(),
                                                          zero_pnts_sign = projection_result['zero_pnts_sign'].detach(),
                                                          correct=correct,
                                                          eps = (eps_correct * correct.float() + eps_wrong * (~correct).float())[projection_result['sign_changed']]
                                                          .view(projection_result['sign_changed'].sum().item(),1,1,1),
                                                          weight = (self.conf.get_float('adversarial_training.correct_weight') * correct.float() + 1.0 * (~correct).float())[projection_result['sign_changed']],
                                                          debug=debug))



