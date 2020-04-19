from torch import nn
import torch
import utils.general as utils
import numpy as np
import torch.nn.functional as F

class MarginLoss(nn.Module):
    def __init__(self,reduction,correct_negative,batch_size,in_adv,clamp=None):
        super().__init__()
        self.clamp = clamp
        self.reduction=reduction
        self.correct_negative = correct_negative
        self.in_adv = in_adv
    def forward(self, input,target):
        if ('output' in str(type(input))):
            input = input.output
        n = input.shape[0]
        outputs_true = input[torch.arange(n, device=input.device), target]
        rups_id = self.get_runner_up_id(input, target)
        rups = input[torch.arange(n, device=input.device), rups_id]
        diff = (outputs_true - rups)
        diff = (diff.clamp(max=self.clamp) if (self.clamp) else diff)
        diff = -diff if self.correct_negative else diff
        if (self.reduction == 'sum'):
            res = diff.sum()
        elif(self.reduction == 'mean'):
            res = diff.mean()
        elif (self.reduction == 'none'):
            res = diff
        else:
            print ("error")

        return res if self.in_adv else utils.dict_to_nametuple("loss_res",dict(loss=res,debug={}))


    def get_runner_up_id(self,input, target):
        tmp = input.clone().data
        n = target.shape[0]
        tmp[torch.arange(n, device=input.device), target] = -torch.tensor(float('inf'))
        rups_id = tmp.argmax(1)
        return rups_id

class TradesLoss(nn.Module):
    def __init__(self,batch_size,reduction,in_adv,beta):
        super().__init__()
        self.batch_size = batch_size
        self.reduction=reduction
        self.beta = beta

    def forward(self, input,target):
        debug = {}
        criterion_kl = nn.KLDivLoss(size_average=False)

        loss_natural = F.cross_entropy(input.loss_natural_logits, target)

        loss_robust = (1.0 / self.batch_size) * criterion_kl(F.log_softmax(input.logits_adv, dim=1),
                                                        F.softmax(input.logits, dim=1))
        loss = loss_natural + self.beta * loss_robust
        return utils.dict_to_nametuple('loss_res',dict(loss=loss ,debug=debug))

class AdvFinalDistanceLoss(nn.Module):
    def __init__(self,batch_size,reduction,in_adv,is_star=True):
        super().__init__()

        self.batch_size = batch_size
        self.reduction=reduction
        self.is_star = is_star

    def forward(self, input,target):
        loss = utils.get_cuda_ifavailable(torch.tensor(0.0))

        if not input.zero_pnts_sample_network is None:
            if (self.is_star):
                d = utils.distance_star(input.effective_samples,input.zero_pnts_sample_network,input.zero_pnts_grad)
            else:
                d = torch.norm(input.effective_samples - input.zero_pnts_sample_network,p=np.inf,dim=[1,2,3])
            loss = input.weight * (input.eps.flatten() + input.zero_pnts_sign.flatten() * d).clamp_min(0)
            loss = loss.sum()

        print (self.reduction == 'mean')
        return utils.dict_to_nametuple('loss_res',dict(loss=(loss/self.batch_size if self.reduction == 'mean' else loss),debug={}))



class Xent(torch.nn.CrossEntropyLoss):
    def __init__(self,batch_size,reduction,in_adv,correct_negative=None):

        super().__init__(reduction=reduction)
        self.in_adv = in_adv
    def forward(self, input,target):
        loss_res = super().forward(input.output,target)  if ('output' in str(type(input))) else super().forward(input, target)
        return loss_res if self.in_adv else utils.dict_to_nametuple('loss_res',dict(loss=loss_res,debug={}))

class ReconDistanceLoss(nn.Module):
    def __init__(self, loss_lambda):
        super().__init__()
        self.loss_lambda = loss_lambda

    def forward(self, zerolevelset_points,
                genlevelset_points,
                pc_input,
                zerolevelset_eval,
                gen_points_eval,
                manifold_pnts_pred,
                loss_lambda):

        if not zerolevelset_points is None:
            proj_pnts = torch.cat([zerolevelset_points, genlevelset_points], dim=0)
            proj_eval = torch.cat([zerolevelset_eval,gen_points_eval],dim=0)
        else:
            proj_pnts = genlevelset_points
            proj_eval = gen_points_eval
        dist_matrix = utils.get_dist_matrix(proj_pnts,pc_input)
        dist_to_pc = dist_matrix.min(dim=1)[0]
        first_term = torch.abs(torch.sqrt(dist_to_pc.abs() + 1.0e-7) - proj_eval.abs().sum(dim=-1))

        second_term = manifold_pnts_pred.abs().sum(dim=-1)


        if (loss_lambda is None):
            loss_lambda = self.loss_lambda
        loss = first_term.mean() + loss_lambda*second_term.mean()
        return {"loss":loss,'first_term':first_term.mean(),'second_term':second_term.mean()}




