import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, Dataset
import pandas as pd
from PIL import Image
import pickle as pk
import os

from sklearn.metrics import confusion_matrix

import numpy as np



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