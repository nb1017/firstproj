import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from utils import *
from models.VGGNet import VGG
import torchvision
import torchvision.transforms as transforms
import os
from tensorboardX import SummaryWriter

def train(epochs,best_acc=0.):
    criterion=nn.CrossEntropyLoss()
    optimizer=optim.Adam(net.parameters(), lr=0.001, betas=[0.5,0.999])
    print_every=20

    Loss=[]

    for epoch in range(epochs):
        print('Epoch : %d/%d' % (epoch+1, epochs))
        net.train()
        for i, (images, labels) in enumerate(train_loader):
            images, labels=images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs=net(images)
            loss=criterion(outputs, labels)
            if (i+1)%print_every==0 :
                print('Loss : %.5f' % loss.item())
                writer.add_scalar('clsf/train_loss', loss.item())
                Loss.append(loss.item())
            loss.backward()
            optimizer.step()

        net.eval()
        print('Validation starts!')
        total=0
        correct=0
        for i, (images, labels) in enumerate(valid_loader):
            images, labels=images.to(device), labels.to(device)
            outputs=net(images)
            loss=criterion(outputs,labels)
            _,predicted=torch.max(outputs.data,1)
            correct+=torch.sum(predicted==labels.data)
            total+=labels.size(0)
            if (i + 1) % print_every == 0:
                writer.add_scalar('clsf/validation_loss', loss.item())
                print('Loss : ',loss.item())

        acc=correct.item()/total
        print('Acc : %.4f' % acc)
        if acc>best_acc:
            best_acc=acc
            if not os.path.exists(ckpt_dir):
                os.mkdir(ckpt_dir)
                torch.save({
                    'net' : net.state_dict(),
                    'acc' : acc},
                    os.path.join(ckpt_dir,'ckpt.7'))
            else :
                torch.save({
                    'net' : net.state_dict(),
                    'acc' : acc}, os.path.join(ckpt_dir,'ckpt.7'))


def test():
    print_every=20
    criterion=nn.CrossEntropyLoss()
    net.eval()
    print('Test starts!')
    total = 0
    correct = 0
    for i,(images, labels) in enumerate(valid_loader):
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        loss = criterion(outputs, labels)
        _, predicted = torch.max(outputs.data, 1)
        correct += torch.sum(predicted == labels.data)
        total += labels.size(0)
        if (i + 1) % print_every == 0:
            print('Loss : ', loss.item())
    acc = correct.item() / total
    print('Acc : %.4f' % acc)

if __name__=='__main__':
    writer=SummaryWriter()

    train_set = torchvision.datasets.CIFAR10('./data', train=True, transform=transforms.ToTensor(), download=True)
    mean, std = get_mean_and_std(train_set)
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation([0.5, 15]),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    train_set = torchvision.datasets.CIFAR10('./data', train=True, transform=train_transform, download=True)
    test_set = torchvision.datasets.CIFAR10('./data', train=False, transform=test_transform, download=True)
    valid_set = torchvision.datasets.CIFAR10('./data', train=True, transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]))

    train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, batch_size=32, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_set, shuffle=False, batch_size=32, num_workers=4)
    valid_loader = torch.utils.data.DataLoader(valid_set, shuffle=True, batch_size=32, num_workers=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt_dir = './ckpt'
    print('Start ...')
    net = VGG('VGG16')
    net = net.to(device)
    best_acc=0.

    if os.path.exists(os.path.join(ckpt_dir,'ckpt.7')):
        print('Loading ...')
        loading=torch.load(os.path.join(ckpt_dir,'ckpt.7'))
        net.load_state_dict(loading['net'])
        best_acc=loading['acc']

    train(50,best_acc)
    test()
    writer.close()