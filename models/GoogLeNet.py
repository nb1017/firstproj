import torch
import torch.nn as nn
import torch.nn.functional as F

class Inception(nn.Module):
    def __init__(self, in_planes, n1x1, n3x3r,n3x3,n5x5r, n5x5,pool_planes):
        super(Inception,self).__init__()
        self.b1=nn.Sequential(nn.Conv2d(in_planes,n1x1,kernel_size=1),
                              nn.BatchNorm2d(n1x1),
                              nn.ReLU(True)
                              )
        self.b2=nn.Sequential(nn.Conv2d(in_planes,n3x3r,kernel_size=1),
                              nn.BatchNorm2d(n3x3r),
                              nn.ReLU(True),
                              nn.Conv2d(n3x3r,n3x3, kernel_size=3,padding=1),
                              nn.BatchNorm2d(n3x3),
                              nn.ReLU(True)
                              )
        self.b3=nn.Sequential(nn.Conv2d(in_planes, n5x5r, kernel_size=1),
                              nn.BatchNorm2d(n5x5r),
                              nn.ReLU(True),
                              nn.Conv2d(n5x5r, n5x5, kernel_size=5, padding=2),
                              nn.BatchNorm2d(n5x5),
                              nn.ReLU(True)
                              )
        self.b4=nn.Sequential(nn.MaxPool2d(3,stride=1,padding=1),
                              nn.Conv2d(in_planes, pool_planes,kernel_size=1),
                              nn.BatchNorm2d(pool_planes),
                              nn.ReLU(True)
                              )
    def forward(self, x):
        b1=self.b1(x)
        b2=self.b2(x)
        b3=self.b3(x)
        b4=self.b4(x)
        return torch.cat([b1, b2,b3,b4],1)

class GoogleNet(nn.Module):
    def __init__(self):
        super(GoogleNet,self).__init__()
        self.pre_layers=nn.Sequential(
            nn.Conv2d(3,192,kernel_size=3,padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(True)
        )
        self.a3=Inception(192,64,96,128,16,32,32)
        self.b3=Inception(256,128,128,192,32,96,64)
        self.maxpool=nn.MaxPool2d(kernel_size=3,stride=2)

        self.a4=Inception(480,192,96,208,16,48,64)
        self.b4=Inception(512, 160,112,224,24,64,64)
        self.c4=Inception(512,128,128,256,24,64,64)
        self.d4=Inception(512,112,144,288,32,64,64)
        self.e4=Inception(528,256,160,320,32,128,128)

        self.a5=Inception(832,256,160,320,32,128,128)
        self.b5=Inception(832,384,192,384,48,128,128)

        self.avgpool=nn.AvgPool2d(7,stride=1)
        self.linear=nn.Linear(1024,10)
    def forward(self,x):
        out=self.pre_layers(x)
        out=self.a3(out)
        out=self.b3(out)
        out=self.maxpool(out)
        out=self.a4(out)
        out=self.b4(out)
        out=self.c4(out)
        out=self.d4(out)
        out=self.e4(out)
        out=self.maxpool(out)
        out=self.a5(out)
        out=self.b5(out)
        out=self.avgpool(out)
        out=out.view(out.size(0),-1)
        out=F.dropout(self.linear(out), p=0.4)
        return out

def test():
    net=GoogleNet()
    x=torch.randn(2,3,32,32)
    y=net(x)
    print(y.size())

if __name__=='__main__':
    test()