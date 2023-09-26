import torch
import torchvision
from torchvision.models import resnet18,ResNet18_Weights
import torchvision.transforms as transforms
from torchvision import datasets, models
from torch.utils.data import DataLoader, Subset, Dataset
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
import glob
import argparse
import os
import pickle as pk



class OPT:

    seed = 42
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    
    # Data
    num_workers = 4
    batch_size = 256
    epochs = 300
    
    lr = 0.001
    wd = 5e-4
    momentum = 0.9
    data = '/home/jb/data' 
    model_weights_name = '/home/jb/Documents/MachineUnlearning/weights/net_weights_resnet18_cifar100'
    outputs_from_model = '/home/jb/Documents/MachineUnlearning/MIA_data/net_weights_resnet18_cifar100'


def train_model(model,optimizer,criterion,scheduler,trainloader,testloader,opt,args,):
    if args.test:
        print('In test: ')
        model.load_state_dict(torch.load(opt.model_weights_name+'.pth'))
        acc_test,cm = compute_accuracy(model, testloader,opt)
        print(f'test_acc: {acc_test}')

    else:
        acc_best = 0
        for epoch in range(opt.epochs):  # loop over the dataset multiple times
            
            model.train()
            running_loss = 0.0

            for i, data in enumerate(trainloader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(opt.device), labels.to(opt.device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
            
            scheduler.step()
            acc_train,_ = compute_accuracy(model, trainloader,opt)
            acc_test,cm = compute_accuracy(model,testloader ,opt)
            print(f'epoch_{round(epoch,3)}, train_acc: {round(acc_train,3)}, test_acc: {round(acc_test,3)}, loss: {round(running_loss/i,3)}')

            if acc_test > acc_best:
                acc_best = acc_test
                torch.save(model.state_dict(), opt.model_weights_name+'.pth')

        print('Finished Training')
        ##### load best model  ###
        model.load_state_dict(torch.load(opt.model_weights_name+'.pth'))
        return model



def compute_accuracy(net, loader, opt):
    correct = 0
    total = 0
    net.eval()  # Set the model to evaluation mode
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for data in loader:
            inputs, labels = data
            inputs, labels = inputs.to(opt.device), labels.to(opt.device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Store all labels and predictions for confusion matrix
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)

    return 100 * correct / total,cm

def obtain_MIA_data(model,loader,opt,train=True):
    model.eval()  # Set the model to evaluation mode
    
    classes = []
    predictions = []

    with torch.no_grad():
        for data in loader:
            inputs, labels = data
            inputs, labels = inputs.to(opt.device), labels.to(opt.device)
            outputs = model(inputs)
            
            # Store all labels and predictions for confusion matrix
            classes.extend(labels.detach().cpu().numpy())
            predictions.extend(outputs.detach().cpu().numpy())

    classes = np.array(classes)
    predictions = np.array(predictions)
    if train:
        case = np.ones_like(classes)
    else:
        case = np.zeros_like(classes)
    return classes,predictions,case

mean = {
'cifar10': (0.4914, 0.4822, 0.4465),
'cifar100': (0.5071, 0.4867, 0.4408),
}

std = {
'cifar10': (0.2023, 0.1994, 0.2010),
'cifar100': (0.2675, 0.2565, 0.2761),
}

transform_train = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean['cifar100'], std['cifar100'])
    ])

transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean['cifar100'], std['cifar100'])
    ])


class Custom_cifar100_Dataset(Dataset):
    def __init__(self,root='/home/jb/data', train=True, transform=None,runs=1):
        
        base_folder = "cifar-100-python"
        file_path_train = os.path.join(root, base_folder,'train')
        file_path_test = os.path.join(root, base_folder,'test')
        
        self.data_all = []
        self.targets_all = []
        
        for file_path in [file_path_train,file_path_test]:
            with open(file_path, "rb") as f:
                    entry = pk.load(f, encoding="latin1")
                    self.data_all.append(entry["data"])
                    if "labels" in entry:
                        self.targets_all.extend(entry["labels"])
                    else:
                        self.targets_all.extend(entry["fine_labels"])

        self.data_all = np.vstack(self.data_all).reshape(-1, 3, 32, 32)
        self.data_all = self.data_all.transpose((0, 2, 3, 1))  # convert to HWC
        self.targets_all = np.asarray(self.targets_all)
        np.random.seed(42)

        N = self.targets_all.shape[0]
        for _ in range(runs):
            id_shuffle = np.random.permutation(N)
        self.data_all = self.data_all[id_shuffle]
        self.targets_all = self.targets_all[id_shuffle]

        if train:
            self.data = self.data_all[:int(0.80*N),:,:,:]
            self.targets = self.targets_all[:int(0.80*N)]
        else:
            self.data = self.data_all[int(0.80*N):,:,:,:]
            self.targets = self.targets_all[int(0.80*N):]

        self.transform = transform
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        image = self.data[idx]
        image = Image.fromarray(image)
        label = self.targets[idx]

        if self.transform:
            image = self.transform(image)

        return image, label