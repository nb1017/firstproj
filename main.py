import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from utils import *
from models.VGGNet import VGG
import torchvision
import torchvision.transforms as transforms

if __name__=='__main__':
    def train(epoch):
        Loss = []
        print('\nEpoch : %d' % epoch)
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            Loss.append(loss.item())

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            if batch_idx%100==99:
                print('Loss : {}'.format(loss.item()))
        torch.save('ckpt.pt',{
            'net':net.state_dict(),
            'acc': correct/total,
            'epoch':epoch
        })


    def test():
        global best_acc
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
            print('accuracy : {}'.format(correct/total))


    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    best_acc = 0
    start_epoch = 0
    print('--> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    transform_test=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    trainset=torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform_train,download=True)
    trainloader=DataLoader(dataset=trainset, batch_size=16, shuffle=True, num_workers=0)

    testset=torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform_test,download=True)
    testloader=DataLoader(dataset=testset, batch_size=16,shuffle=False,num_workers=0)

    learning_rate=0.001
    net=VGG('VGG16')
    net=net.to(device)

    try :
        print('>>>> Resuming from checkpoint..')
        checkpoint=torch.load('./checkpoint/ckpt.pt')
        net.load_state_dict(checkpoint['net'])
        best_acc=checkpoint['acc']
        best_epoch=checkpoint['epoch']
    except:
        print('Error: no checkpoint directory found!')


    criterion=nn.CrossEntropyLoss()

    optimizer=optim.Adam(net.parameters(), lr=learning_rate)

    train(5)
    test()