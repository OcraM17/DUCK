import numpy as np
import os
import requests
import torch
from torch.utils.data import DataLoader, Subset
import torchvision
from torchvision import transforms
from opts import OPT as opt

def split_retain_forget(dataset, class_to_remove):

    # find forget indices
    forget_idx = np.where(np.array(dataset.targets) == class_to_remove)[0]
    
    forget_mask = np.zeros(len(dataset.targets), dtype=bool)
    forget_mask[forget_idx] = True
    retain_idx = np.arange(forget_mask.size)[~forget_mask]

    forget_set = Subset(dataset, forget_idx)
    retain_set = Subset(dataset, retain_idx)

    return forget_set, retain_set


def get_dsets_remove_class(class_to_remove):

    # download and pre-process CIFAR10
    normalize = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ]
    )

    # we split held out - train
    train_set = torchvision.datasets.CIFAR10(root=opt.data_path, train=True, download=True, transform=normalize)
    val_set = torchvision.datasets.CIFAR10(root=opt.data_path, train=False, download=True, transform=normalize)
    #val_set, test_set = torch.utils.data.random_split(held_out, [0.7, 0.3])

    val_forget_set, val_retain_set = split_retain_forget(val_set, class_to_remove)
    forget_set, retain_set = split_retain_forget(train_set, class_to_remove)

    # validation set and its subsets 
    all_val_loader = DataLoader(val_set, batch_size=opt.batch_size, shuffle=True, num_workers=2)
    val_fgt_loader = DataLoader(val_forget_set, batch_size=opt.batch_size, shuffle=True, num_workers=2)
    val_retain_loader = DataLoader(val_retain_set, batch_size=opt.batch_size, shuffle=True, num_workers=2)
    
    # all train and its subsets
    all_train_loader = DataLoader(train_set, batch_size=opt.batch_size, shuffle=True, num_workers=2)
    forget_loader = DataLoader(forget_set, batch_size=opt.batch_size, shuffle=True, num_workers=2)
    retain_loader = DataLoader(retain_set, batch_size=opt.batch_size, shuffle=True, num_workers=2)

    # never seen test set (non sblindatelo!!)
    #test_loader = DataLoader(test_set, batch_size=128, shuffle=True, num_workers=2)

    return all_train_loader, forget_loader, retain_loader, val_fgt_loader, val_retain_loader, all_val_loader




def get_dsets(RNG):

    # download and pre-process CIFAR10
    normalize = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    
    # we split held out data into test and validation set
    train_set = torchvision.datasets.CIFAR10(root=opt.data_path, train=True, download=True, transform=normalize)
    train_loader = DataLoader(train_set, batch_size=opt.batch_size, shuffle=True, num_workers=2)

    # we split held out data into test and validation set
    held_out = torchvision.datasets.CIFAR10(root=opt.data_path, train=False, download=True, transform=normalize)
    test_set, val_set = torch.utils.data.random_split(held_out, [0.5, 0.5])
    test_loader = DataLoader(test_set, batch_size=opt.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=opt.batch_size, shuffle=True, num_workers=2)

    # download the forget and retain index split
    local_path = "forget_idx.npy"
    if not os.path.exists(local_path):
        response = requests.get(
            "https://unlearning-challenge.s3.eu-west-1.amazonaws.com/cifar10/" + local_path
        )
        open(local_path, "wb").write(response.content)
    forget_idx = np.load(local_path)

    # construct indices of retain from those of the forget set
    forget_mask = np.zeros(len(train_set.targets), dtype=bool)
    forget_mask[forget_idx] = True
    retain_idx = np.arange(forget_mask.size)[~forget_mask]

    # split train set into a forget and a retain set
    forget_set = torch.utils.data.Subset(train_set, forget_idx)
    retain_set = torch.utils.data.Subset(train_set, retain_idx)

    forget_loader = torch.utils.data.DataLoader(forget_set, batch_size=opt.batch_size, shuffle=True, num_workers=2)
    retain_loader = torch.utils.data.DataLoader(retain_set, batch_size=opt.batch_size, shuffle=True, num_workers=2)


    return train_loader, test_loader, forget_loader, retain_loader

