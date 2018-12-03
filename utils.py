import os
import sys
import time
import math
import shutil
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.utils.data import DataLoader


def get_mean_and_std(dataset):
    dataloader=DataLoader(dataset, batch_size=1,shuffle=True, num_workers=0)
    mean=torch.zeros(3)
    std=torch.zeros(3)

    print('Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i]+=inputs[:,i,:,:].mean()
            std[i]+=inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight,mode='fan_out')
            if m.bias:
                init.constant(m.bias,0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight,1)
            init.constant(m.bias,0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight,std=1e-3)
            if m.bias:
                init.constant(m.bias,0)