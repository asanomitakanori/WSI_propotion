from numpy.core.numeric import NaN
import torch
import torch.utils.data
import numpy as np 


class CrossEntropy(torch.nn.Module):
    ''' 
    CrossEntropy for imbalance dataset
    '''
    def __init__(self):
        super(CrossEntropy, self).__init__()
        self.delta = 1e-5

    def forward(self, input, gts):
        assert len(input.shape) == 4, 'shape is different'
        # for i in range(input.shape[1]):
        #     sum = torch.sum(gts[:, i]) + 1
        #     sum /= propotion
        #     loss += torch.mean(-torch.log(input[:, i]) * gts[:, i] * (1/sum))
        loss = torch.mean(-torch.log(input + self.delta) * gts)
        return loss

class PropotionLoss(torch.nn.Module):
    ''' 
    This code is PropotionLoss
    '''
    def __init__(self):
        super(PropotionLoss, self).__init__()
        self.delta = 1e-5

    def forward(self, input, propotion):
        # output : prediction's propotion
        # gts : ground truth label's propotion
        output = input.sum(dim=0)
        output = output / input.sum()
        gts =  torch.tensor([propotion['0'], 
                             propotion['1'], 
                             propotion['2']
                            ]).to(output.device)
        gts = gts / gts.sum()
        loss = torch.mean(-torch.log(output + self.delta) * gts)
        return loss

class IoU:
    """
    IoU for mnist
    """
    def loss(self, y_true, y_pred):
        eps = 1e-5
        temp = y_true.clone()
        temp2 = y_pred.clone()
        temp[temp>0] = 1
        temp2[temp2>0] = 1
        return torch.sum(temp[temp2==1]==1) / (torch.sum(temp) + torch.sum(temp2) - torch.sum(temp[temp2==1]==1) + eps)
        

class IoU2:
    """
    IoU
    """
    def loss(self, y_true, y_pred):
        eps = 1e-5
        temp = y_true.copy()
        temp2 = y_pred.copy()
        temp[temp>0] = 1
        temp2[temp2>0] = 1
        return np.sum(temp[temp2==1]==1) / (np.sum(temp) + np.sum(temp2) - np.sum(temp[temp2==1]==1) + eps)


class Dice:
    """
    N-D dice for mnist
    """
    def loss(self, y_true, y_pred):

        eps = 1e-5
        temp = y_true.clone()
        temp2 = y_pred.clone()

        temp[temp>0] = 1
        temp2[temp2>0] = 1
        return  2*torch.sum(temp[temp2==1]==1) / (torch.sum(temp) + torch.sum(temp2) + eps)

