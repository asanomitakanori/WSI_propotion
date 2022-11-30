
import torch
from torch import nn
import torchvision.models as models
from Model.model_parts import *


class Resnet(nn.Module):

    def __init__(self, n_classes):
        super(Resnet, self).__init__()
        self.net = models.resnet50(pretrained=True)

        self.net.conv1.stride = 3
        self.net.conv1.in_channels = 3

        layer0 = nn.Sequential(
                               self.net.conv1, 
                               self.net.bn1, 
                               self.net.relu, 
                               self.net.maxpool
                              )
        layer1 = self.net.layer1
        layer2 = self.net.layer2
        # layer3 = self.net.layer3

        Out0 = OutConv(in_channels=layer0[0].out_channels, out_channels=n_classes)
        Out1 = OutConv(in_channels=layer1[-1].conv3.out_channels, out_channels=n_classes)
        Out2 = OutConv(in_channels=layer2[-1].conv3.out_channels, out_channels=n_classes)
        # Out3 = OutConv(in_channels=layer3[-1].conv3.out_channels, out_channels=n_classes)

        self.net_E = nn.Sequential(
            layer0,  #1
            layer1,  #2
            layer2,  #3
            # layer3,  #4
        )
        self.net_OE = nn.Sequential(
            Out0,    #1
            Out1,    #2
            Out2,    #3
            # Out3,    #4
        )
        assert len(self.net_E) == len(self.net_OE), \
            f'Encoder layer nums {len(self.net_E)} is different from Decorder layer nums {len(self.net_OE)}'

    def forward(self, input, layer_num=None):
        if layer_num != None:
            feature = self.net_E[layer_num](input)
            x = self.net_OE[layer_num](feature)
        else:
            feature = self.net_E(input)
            x = self.net_OE[-1](feature)
        return x, feature