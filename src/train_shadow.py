# Import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import Subset,SubsetRandomSampler
from models.allcnn import AllCNN
from torch import nn

from opts import OPT as opt
import os 
from dsets import  get_dsets_shadow

def trainer(nmodel):
    # Initialize the model
    if opt.model == 'resnet18':
        model= torchvision.models.resnet18(pretrained=True).to(opt.device)
    elif opt.model=='resnet34':
        model= torchvision.models.resnet34(pretrained=True).to(opt.device)
    elif opt.model=='resnet50':
        model = torchvision.models.resnet50(pretrained=True).to(opt.device)
    elif opt.model=='AllCNN':
        N=32
        if opt.dataset == 'tinyImagenet':
            N=64
        model = AllCNN(n_channels=3, num_classes=opt.num_classes,img_size=N).to(opt.device)
        
    
    if opt.dataset == 'cifar10':
        os.makedirs(f'./weights_shadow/chks_cifar10/{opt.mode}', exist_ok=True)
        # Load CIFAR-10 data
        model.fc = nn.Linear(512, opt.num_classes).to(opt.device)


    elif opt.dataset == 'cifar100':
        os.makedirs(f'./weights_shadow/chks_cifar100/{opt.mode}', exist_ok=True)
        if 'resnet' in opt.model:    
            model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False).to(opt.device)
            model.maxpool = nn.Identity()
            model.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(model.fc.in_features, opt.num_classes)).to(opt.device)

    elif opt.dataset == 'tinyImagenet':
        #dataloader
        os.makedirs(f'./weights_shadow/chks_tinyImagenet/{opt.mode}', exist_ok=True)
        if 'resnet' in opt.model:
            model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False).to(opt.device)
            model.maxpool = nn.Identity()
            model.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(model.fc.in_features, opt.num_classes)).to(opt.device)
    
    #dataloader
    if opt.mode == "CR":
        #the test set is composed by retain classes
        forget_set, retain_set, test_forget_set, test_set = get_dsets_shadow(opt.class_to_remove)
    
    else:
        file_fgt = f'{opt.root_folder}forget_id_files/forget_idx_{opt.seed}_{opt.dataset}_seed_{i}.txt'
        forget_set, retain_set, test_set = get_dsets_shadow(file_fgt=file_fgt)

    
    indices = np.random.choice(len(retain_set), size=len(retain_set)//2, replace=False)

    # Create a SubsetRandomSampler with the random indices
    sampler = SubsetRandomSampler(indices)

    trainloader = torch.utils.data.DataLoader(retain_set, batch_size=256, shuffle=True, num_workers=opt.num_workers,sampler=sampler)
    testloader = torch.utils.data.DataLoader(test_set, batch_size=256, shuffle=False, num_workers=opt.num_workers)

    epochs=100
    criterion = nn.CrossEntropyLoss(label_smoothing=0.4)
    optimizer = optim.SGD(model.parameters(), lr=0.1, weight_decay=5e-5)
    train_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)


    # Train the network
    best_acc = 0.0
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        model.train()
        correct, total = 0, 0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(opt.device), labels.to(opt.device)
            optimizer.zero_grad()
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_acc = 100 * correct / total
        train_loss = running_loss / len(trainloader)
        train_scheduler.step()
        torch.save(model.state_dict(), f'weights_shadow/chks_{opt.dataset}/{opt.mode}/best_checkpoint_{opt.model}_test_numModel_{nmodel}.pth')

        if epoch % 5 == 0:        
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for data in testloader:
                    inputs, labels = data
                    inputs, labels = inputs.to(opt.device), labels.to(opt.device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            val_acc = 100 * correct / total
            if val_acc > best_acc:
                best_acc = val_acc
            print('Epoch: %d, Train Loss: %.3f, Train Acc: %.3f, Val Acc: %.3f, Best Acc: %.3f' % (epoch, train_loss, train_acc, val_acc, best_acc))
    return best_acc


if __name__ == '__main__':
    for nmodel in range(40):
        trainer(nmodel)