import torch
import numpy as np
import utils.general as utils
import abc

class Sampler(metaclass=abc.ABCMeta):
    def __init__(self,sigmas,points_fraction):
        self.points_fraction = points_fraction
        self.num_points_per_sample = len(sigmas)
        self.sigmas = sigmas

    @abc.abstractmethod
    def get_points(self,pc_input):
        pass

    def get_sampler(sampler_type):
        return utils.get_class("model.common.sample.{0}".format(sampler_type))


class NormalAroundPoint(Sampler):
    def get_points(self,pc_input):
        dim = pc_input.shape[-1]

        dists = [torch.distributions.MultivariateNormal(utils.get_cuda_ifavailable(torch.zeros(dim)),
                                                      utils.get_cuda_ifavailable(torch.diag(torch.Tensor([sigma**2]*dim))))
                 for sigma in self.sigmas]
        # return (pc_input.unsqueeze(1).repeat(1,self.num_points_per_sample,1) +
        #         torch.cat([dist.sample([pc_input.shape[0],1]) for dist in dists],dim=1)).reshape(-1,dim)
        return pc_input.repeat(self.num_points_per_sample,1) + torch.cat([dist.sample([pc_input.shape[0]]) for dist in dists],dim=0)

class NormalAroundPointWithUniform(NormalAroundPoint):
    def get_points(self,pc_input):
        num_points = int(self.points_fraction * pc_input.shape[0])
        idx = np.random.choice(np.arange(pc_input.shape[0]),num_points,False)
        normal_around_points = super().get_points(pc_input[idx])
        return torch.cat([normal_around_points,
                          utils.get_cuda_ifavailable(torch.empty(pc_input.shape[0] - num_points,pc_input.shape[1]).uniform_(-1.1,1.1))],dim=0)


