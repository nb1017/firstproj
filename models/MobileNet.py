import torch
import torch.nn as nn
import torch.nn.functional as F

class condwcon(nn.Module):
    def __init__(self, inplane, outplane, stride=1):
        super(condwcon, self).__init__()
        self.convdw = nn.Conv2d(inplane, inplane, kernel_size=3, stride=stride, padding=1, groups=inplane)
        self.bn1 = nn.BatchNorm2d(inplane)
        self.conv = nn.Conv2d(inplane, outplane, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(outplane)

    def forward(self, x):
        x = F.relu(self.bn1(self.convdw(x)))
        x = F.relu(self.bn2(self.conv(x)))
        return x


class MobileNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(MobileNet, self).__init__()
        self.inplane = 32
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.layer1 = self._make_layers([1, 2, 1, 2, 1, 2], [64, 128, 128, 256, 256, 512])
        self.layer2 = self._make_layers([1, 1, 1, 1, 1], [512, 512, 512, 512, 512])
        self.layer3 = self._make_layers([2, 1], [1024, 1024])
        self.avg = nn.MaxPool2d(7)
        self.fc = nn.Linear(1024, num_classes)

    def _make_layers(self, stride, planes):
        layers = []
        for st, p in zip(stride, planes):
            layers += [condwcon(self.inplane, p, stride=st)]
            self.inplane = p
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x