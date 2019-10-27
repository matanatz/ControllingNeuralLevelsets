import torch.nn as nn
import torch

class SampleNetwork(nn.Module):
    def forward(self, network,levelset_points,levelset_points_Dx,network_eval_on_levelset_points,target=None,additional_func=None):

        network_eval = network.forward(levelset_points)#.squeeze()
        if (not additional_func is None):
            network_eval = additional_func(network_eval,target=target)

        sum_square_grad = torch.sum(levelset_points_Dx ** 2, dim=list(range(len(levelset_points_Dx.shape)))[1:], keepdim=True)
        sample_layer = levelset_points - (
                network_eval - network_eval_on_levelset_points).view([network_eval.shape[0]] + [1]*(len(levelset_points_Dx.shape) - 1)) * (
                               levelset_points_Dx / sum_square_grad.clamp_min(1.0e-6))
        return sample_layer

