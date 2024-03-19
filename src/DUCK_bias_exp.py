from copy import deepcopy
from dsets import get_dsets_remove_class, get_dsets
import pandas as pd

from utils import accuracy, set_seed, get_retrained_model,get_trained_model

from opts import OPT as opt
import torch.nn as nn
from tqdm import tqdm
import os
import torch
from torch.utils.data import DataLoader, Subset

import json
import shap
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import numpy as np
from utils import set_seed
from torch import nn 
from Unlearning_methods import choose_method
from utils import set_seed

set_seed(42)
class Poisoned_CIFAR10(torch.utils.data.Dataset):
    def __init__(self, transform,get_id=False):
        # Load the CIFAR10 dataset
        cifar10 = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
        self.imgs=cifar10.data
        self.labels=cifar10.targets
        self.car_idx=1
        self.truck_idx=9
        # self.poisoned_indexes_car=np.random.choice(5000, 200, replace=False)+(self.car_idx*5000)
        # self.poisoned_indexes_truck=np.random.choice(5000, 200, replace=False)+(self.truck_idx*5000)
        self.poisoned_indexes_car=(torch.tensor(self.labels)==self.car_idx).nonzero()[np.random.choice(5000, 200, replace=False)]
        self.poisoned_indexes_truck=(torch.tensor(self.labels)==self.truck_idx).nonzero()[np.random.choice(5000, 200, replace=False)]
        self.transform=transform
        self.get_id = get_id

    def save_idxs(self,PATH):
        np.save(PATH, np.concatenate((self.poisoned_indexes_car,self.poisoned_indexes_truck)))
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img=self.imgs[idx]
        label=self.labels[idx]
        id = True
        if idx in self.poisoned_indexes_car:# or idx in self.poisoned_indexes_truck:
            #the image is 32x32 make the upper right corner 8x8 red
            img[0:4,28:32,0]=255
            img[0:4,28:32,1]=0
            img[0:4,28:32,2]=0
            
        if idx in self.poisoned_indexes_car:
            label=self.truck_idx
            id=False
        if idx in self.poisoned_indexes_truck:
            id=True

        img=self.transform(img)
        return img, label
def accuracy(net, loader,fgt=False):
    """Return accuracy on a dataset given by the data loader."""
    correct = 0
    total = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(opt.device), targets.to(opt.device)
        if fgt:
            targets[:] = 1
        outputs = net(inputs)
        _, predicted = outputs.max(1)
        if fgt: print(predicted)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    return correct / total



    
    
original_pretr_model = get_trained_model()
weights_path = '/home/jb/Documents/MachineUnlearning/poisoned_model2.pth'#f'/home/jb/Documents/MachineUnlearning/poison/poisoned_model.pth'
original_pretr_model.load_state_dict(torch.load(weights_path))
un_model = deepcopy(original_pretr_model)
original_pretr_model.to(opt.device)
original_pretr_model.eval()

transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                          torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
dataset=Poisoned_CIFAR10(transform)
### select subset
#load npy indexes
idxs = dataset.poisoned_indexes_car.numpy().astype(np.int64)#np.load('/home/jb/Documents/MachineUnlearning/poison/poisoned_indexes.npy').astype(np.int64)[:200]
forget_mask = np.zeros(len(dataset.imgs), dtype=bool)
forget_mask[idxs] = True

fgt_idx =np.arange(forget_mask.size)[forget_mask]
retain_idx = np.arange(forget_mask.size)[~forget_mask]
forget_set = Subset(dataset, fgt_idx)
retain_set = Subset(dataset, retain_idx)
fgt_loader=torch.utils.data.DataLoader(forget_set, batch_size=1024, shuffle=False)
retain_loader=torch.utils.data.DataLoader(retain_set, batch_size=1024, shuffle=True)

test_set=torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]))
test_loader=torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=True)

print(accuracy(original_pretr_model.eval(),fgt_loader,fgt=True))

###
opt.target_accuracy = 0.01
approach = choose_method(opt.method)(original_pretr_model,retain_loader, fgt_loader,test_loader, class_to_remove=None)
unlearned_model = approach.run()


print(accuracy(unlearned_model.eval(),fgt_loader,fgt=True))
print(accuracy(unlearned_model.eval(),retain_loader))
torch.save(unlearned_model.state_dict(), f"model_xai_test.pth")