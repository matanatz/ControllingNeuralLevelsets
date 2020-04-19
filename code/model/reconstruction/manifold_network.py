import utils.general as utils
import torch.nn as nn
import torch
from model.common.sample import Sampler
from model.common.sample_network import SampleNetwork
import numpy as np

class Decoder(nn.Module):
    def __init__(
        self,
        dims,
        latent_in=(),
        xyz_in_all=None,
        activation=None,
        output_dim = 1
    ):
        super().__init__()

        latent_size = 0
        dims = [latent_size + 3] + dims + [output_dim]

        self.num_layers = len(dims)
        self.latent_in = latent_in
        self.xyz_in_all = xyz_in_all

        for l in range(0, self.num_layers - 1):
            if l + 1 in latent_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]
                if self.xyz_in_all and l != self.num_layers - 2:
                    out_dim -= 3

            lin = nn.Conv1d(dims[l], out_dim,1)

            if l == self.num_layers - 2:
                torch.nn.init.normal_(lin.weight, mean=2 * np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                torch.nn.init.constant_(lin.bias, -1)
            else:
                torch.nn.init.constant_(lin.bias,0)
                torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
            setattr(self, "lin" + str(l), lin)

        self.use_activation = not activation == 'None'
        if self.use_activation:
            self.last_activation = utils.get_class(activation)()
        self.relu = nn.ReLU()


    # input: N x (L+3)
    def forward(self, input):
        input = input.unsqueeze(0).transpose(1,2)
        x = input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))
            if l in self.latent_in:
                x = torch.cat([x, input], 1)
            elif l != 0 and self.xyz_in_all:
                x = torch.cat([x, input], 1)
            x = lin(x)

            if l < self.num_layers - 2:
                x = self.relu(x)

        if self.use_activation:
            x = self.last_activation(x)

        return x[0].transpose(0,1)


class ManifoldNetwork(nn.Module):
    def __init__(self,conf):
        super().__init__()
        self.decoder = Decoder(**conf.get_config('decoder'))
        self.points_projected_init_sampler = Sampler.get_sampler(conf.get_string('zero_level_set_projection.sampler.sampler_type'))(**conf.get_config('zero_level_set_projection.sampler.properties'))
        self.zero_levelset_projection = utils.get_class(
            conf.get_string('zero_level_set_projection.projection.projection_type'))(
            **conf.get_config('zero_level_set_projection.projection.properties'))
        if (len(conf.get_config('general_projection')) > 0):
            self.general_points_sampler = Sampler.get_sampler(conf.get_string('general_projection.sampler.sampler_type'))(**conf.get_config('general_projection.sampler.properties'))
            self.gen_levelset_projection = utils.get_class(
                conf.get_string('general_projection.projection.projection_type'))(**conf.get_config('general_projection.projection.properties'))
        else:
            self.general_points_sampler = None
            self.gen_levelset_projection = None
        self.sample_network = SampleNetwork()

    def forward(self, input):

        manifold_pnts_pred = self.decoder(input)

        if self.general_points_sampler:
            denom = 2
        else:
            denom = 1


        idx = np.random.choice(np.arange(input.shape[0]), input.shape[0]//denom, False)
        other_idx = np.array(list(set(range(input.shape[0])).difference(set(idx.tolist()))))
        points_init = self.points_projected_init_sampler.get_points(input[idx])

        zerolevelset_proj_result = self.zero_levelset_projection.project_points(points_init=points_init,
                                                                             model=self.decoder,
                                                                             latent=None)
        zerolevelset_sample_layer = self.sample_network(self.decoder,**zerolevelset_proj_result)

        if self.general_points_sampler:
            general_points_init = self.general_points_sampler.get_points(input[other_idx])
            gen_proj_result = self.gen_levelset_projection.project_points(points_init=general_points_init,
                                                                                   model=self.decoder,
                                                                                   latent=None)
            genlevelset_sample_layer = self.sample_network(self.decoder,**gen_proj_result)
        else:
            genlevelset_sample_layer = torch.tensor([]).cuda()
            gen_proj_result = {k:torch.tensor([]).cuda() for k in zerolevelset_proj_result.keys()}

        return {"manifold_pnts_pred":manifold_pnts_pred,
                "zerolevelset_sample_layer":zerolevelset_sample_layer,
                "genlevelset_sample_layer":genlevelset_sample_layer,
                "zerolevelset_proj_result": zerolevelset_proj_result,
                "genlevelset_proj_result":gen_proj_result}
