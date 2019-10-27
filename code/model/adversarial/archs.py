import utils.general as utils
import torch.nn as nn
import torch
import numpy as np
from collections import OrderedDict

class ConvNet4(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            torch.nn.Flatten(),
            nn.Linear(64 * 8 * 8, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / n))
                m.bias.data.zero_()
        nn.init.constant_(self.model[-1].bias, 0)

    def forward(self, input,**kwargs):
        x = self.model(input)
        return x

class SmallCNN(nn.Module):
    def __init__(self):
        super(SmallCNN, self).__init__()

        self.num_channels = 1
        self.num_labels = 10

        activ = nn.ReLU(True)

        self.model = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(self.num_channels, 32, 3)),
            ('relu1', activ),
            ('conv2', nn.Conv2d(32, 32, 3)),
            ('relu2', activ),
            ('maxpool1', nn.MaxPool2d(2, 2)),
            ('conv3', nn.Conv2d(32, 64, 3)),
            ('relu3', activ),
            ('conv4', nn.Conv2d(64, 64, 3)),
            ('relu4', activ),
            ('maxpool2', nn.MaxPool2d(2, 2)),
            ('flatten',nn.Flatten()),
            ('fc1', nn.Linear(64 * 4 * 4, 200)),
            ('relu1', activ),
            ('fc2', nn.Linear(200, 200)),
            ('relu2', activ),
            ('fc3', nn.Linear(200, self.num_labels)),
        ]))

        for m in self.modules():
            if isinstance(m, (nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # nn.init.constant_(self.model.fc3.bias, -0.1)
        # nn.init.constant_(self.model.fc2.bias, 0)
        # nn.init.constant_(self.model.fc1.bias, 0)


    def forward(self, input,**kwargs):
        logits = self.model(input)
        return  logits

class WrapNetwork(nn.Module):
    def __init__(
        self,conf
    ):
        super().__init__()

        self.model = utils.get_class(conf.get_string('model'))()

    # input: N x (L+3)
    def forward(self, input,**kwargs):
        output = self.model(input)
        return utils.dict_to_nametuple("output", dict(output=output,
                                                          debug={}))

    def get_correct(self,network_output,target):
        if ('output' in str(type(network_output))):
            pred = network_output.output.argmax(dim=1)  # get the index of the max log-probability
        else:
            pred = network_output.argmax(dim=1)  # get the index of the max log-probability
        correct = pred.eq(target.view_as(pred))
        return correct
