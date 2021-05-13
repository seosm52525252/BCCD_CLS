import os
import torch
import torchvision
import torchvision.transforms as transforms

def dataset(path):
    
    train_path = path + '/train/'
    #test_path = path + '/test/'
    val_path = path + '/validation/'
    
    #ResNet Normal
    transform_train = transforms.Compose([
        transforms.Resize((195,195),),
        #transforms.RandomCrop(30, padding=4),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    
    transform_test = transforms.Compose([
    transforms.Resize((195,195),),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    
    
    trainset = torchvision.datasets.ImageFolder(root= train_path , transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=80, shuffle=True, num_workers=4)
    
    valset = torchvision.datasets.ImageFolder(root= val_path , transform=transform_train)
    valloader = torch.utils.data.DataLoader(trainset, batch_size=40, shuffle=True, num_workers=4)
    
    return trainset, valset, trainloader, valloader
    