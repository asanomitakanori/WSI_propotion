from ast import increment_lineno
import torch
import torch.nn as nn
from Model.model_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, num=16, bilinear=True):
        super().__init__()

        self.inc = OneConv(n_channels, num)
        self.down1 = Down(num, num*2)
        factor = 2 if bilinear else 1
        self.down2 = Down(num*2, num*4)
        self.down3 = Down(num*4, num*8)
        self.up1 = Up(num*12,  num*4 // factor, bilinear)
        self.up2 = Up(num*4,  num*2 // factor, bilinear)
        self.up3 = Up(num*2,  num, bilinear)
        self.out = OutConv(num, n_classes)

    def forward(self, input):
        x1 = self.inc(input)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.down3(x3)
        x = self.up1(x, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.out(x)
        return x
