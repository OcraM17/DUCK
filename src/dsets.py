import numpy as np
import os
import requests
import torch
from torch.utils.data import DataLoader, Subset, random_split,Dataset
import torchvision
from torchvision import transforms
from opts import OPT as opt
from PIL import Image
import pandas as pd
import glob


def split_retain_forget(dataset, class_to_remove):

    # find forget indices
    #print(np.unique(np.array(dataset.targets),return_counts=True))
    if type(class_to_remove) is list:
        forget_idx = None
        for class_rm in class_to_remove:
            if forget_idx is None:
                forget_idx = np.where(np.array(dataset.targets) == class_rm)[0]
            else:
                forget_idx = np.concatenate((forget_idx, np.where(np.array(dataset.targets) == class_rm)[0]))
            
            #print(class_rm,forget_idx.shape)
    else:
        forget_idx = np.where(np.array(dataset.targets) == class_to_remove)[0]

    forget_mask = np.zeros(len(dataset.targets), dtype=bool)
    forget_mask[forget_idx] = True
    retain_idx = np.arange(forget_mask.size)[~forget_mask]

    forget_set = Subset(dataset, forget_idx)
    retain_set = Subset(dataset, retain_idx)

    return forget_set, retain_set


def get_dsets_remove_class(class_to_remove):
    mean = {
            'cifar10': (0.4914, 0.4822, 0.4465),
            'cifar100': (0.5071, 0.4867, 0.4408),
            'tinyImagenet': (0.485, 0.456, 0.406),
            'VGG':(0.547, 0.460, 0.404)
            }

    std = {
            'cifar10': (0.2023, 0.1994, 0.2010),
            'cifar100': (0.2675, 0.2565, 0.2761),
            'tinyImagenet': (0.229, 0.224, 0.225),
            'VGG':(0.323, 0.298, 0.263)
            }

    
    transform_dset_list = [
            transforms.ToTensor(),
            transforms.Normalize(mean[opt.dataset],std[opt.dataset]),
        ]

    transform_dset2 = transforms.Compose(
        [   transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(64, padding=8),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean[opt.dataset],std[opt.dataset]),
        ]
    )
    if opt.model ==  'ViT':
        transform_dset_list.insert(2,transforms.Resize(224,antialias=True))
    transform_dset = transforms.Compose(transform_dset_list)
    # we split held out - train
    if opt.dataset == 'cifar10':
        train_set = torchvision.datasets.CIFAR10(root=opt.data_path, train=True, download=True, transform=transform_dset)
        test_set = torchvision.datasets.CIFAR10(root=opt.data_path, train=False, download=True, transform=transform_dset)
    elif opt.dataset == 'cifar100':
        train_set = torchvision.datasets.CIFAR100(root=opt.data_path, train=True, download=True, transform=transform_dset)
        test_set = torchvision.datasets.CIFAR100(root=opt.data_path, train=False, download=True, transform=transform_dset)
        
    elif opt.dataset == 'tinyImagenet':
        train_set = torchvision.datasets.ImageFolder(root=opt.data_path+'/tiny-imagenet-200/train',transform=transform_dset)
        test_set = torchvision.datasets.ImageFolder(root=opt.data_path+'/tiny-imagenet-200/val/images',transform=transform_dset)

    elif opt.dataset == 'VGG':
        
        #### FIX and uniform to the rest 

        # Load and transform the data
        transform_dset = transforms.Compose([
            transforms.Resize((128,128)),
            transforms.ToTensor(),
            transforms.Normalize(mean[opt.dataset],std[opt.dataset]),
            ])

        data_path = opt.data_path+'/VGG-Face2/data/train/'
        txt_path = opt.data_path+'/VGG-Face2/data/train_list.txt'

        folder_list = glob.glob(os.path.expanduser(data_path)+'*')

        dict_subj = {}
        for fold in folder_list:

            files = glob.glob(fold+'/*.jpg')
            
            if len(files)>500:
                dict_subj[fold.split('/')[-1]] = len(files)
        print(f'Num subject suitable: {len(list(dict_subj.keys()))}')

        df = pd.read_csv(txt_path,sep=',',header=None, names=['Id',])

        sorted_dict_subj = sorted(dict_subj.items(), key=lambda x:x[1], reverse=True)
        sorted_dict_subj = dict(sorted_dict_subj)
        best_10_subject=[]
        skip = 100
        for key in sorted_dict_subj.keys():
            if key!=skip:
                best_10_subject.append(key)
                if len(best_10_subject)==10:
                    break

        #filter for subjects
        mask = df.Id.apply(lambda x: any(item for item in best_10_subject if item in x))
        df = df[mask]
        #shuffle dataframe
        df = df.sample(frac=1)
        
        trainset = CustomDataset_10subj(df,path = data_path,best_10_subject=best_10_subject, train= True, transform=transform_dset)
        all_train_loader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True,num_workers=opt.num_workers)

        testset = CustomDataset_10subj(df,path = data_path,best_10_subject=best_10_subject, train=False, transform=transform_dset)
        all_test_loader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False,num_workers=opt.num_workers)
        
        retain_set = CustomDataset_10subj(df,path = data_path,best_10_subject=best_10_subject, train= True,split=True,retain=True, transform=transform_dset,class_to_remove=class_to_remove)
        forget_set = CustomDataset_10subj(df,path = data_path,best_10_subject=best_10_subject, train= True,split=True,retain=False, transform=transform_dset,class_to_remove=class_to_remove)

        train_fgt_loader = DataLoader(forget_set, batch_size=opt.batch_size, drop_last=False, shuffle=True, num_workers=opt.num_workers)
        train_retain_loader = DataLoader(retain_set, batch_size=opt.batch_size, drop_last=False, shuffle=True, num_workers=opt.num_workers)
        

        testset_retain = CustomDataset_10subj(df,path = data_path,best_10_subject=best_10_subject, train=False,split=True,retain=True,transform=transform_dset,class_to_remove=class_to_remove)
        test_retain_loader = torch.utils.data.DataLoader(testset_retain, batch_size=opt.batch_size, shuffle=False,num_workers=opt.num_workers)

        testset_forget = CustomDataset_10subj(df,path = data_path,best_10_subject=best_10_subject, train=False,split=True,retain=False,transform=transform_dset,class_to_remove=class_to_remove)
        test_fgt_loader = torch.utils.data.DataLoader(testset_forget, batch_size=opt.batch_size, shuffle=False,num_workers=opt.num_workers)

        return all_train_loader,all_test_loader, train_fgt_loader, train_retain_loader, test_fgt_loader, test_retain_loader
    
    test_forget_set, test_retain_set = split_retain_forget(test_set, class_to_remove)
    forget_set, retain_set = split_retain_forget(train_set, class_to_remove)

    # validation set and its subsets 
    all_test_loader = DataLoader(test_set, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)
    test_fgt_loader = DataLoader(test_forget_set, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)
    test_retain_loader = DataLoader(test_retain_set, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)
    
    # all train and its subsets
    all_train_loader = DataLoader(train_set, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)
    train_fgt_loader = DataLoader(forget_set, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)
    train_retain_loader = DataLoader(retain_set, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)


    return all_train_loader,all_test_loader, train_fgt_loader, train_retain_loader, test_fgt_loader, test_retain_loader




def get_dsets(file_fgt=None):
    mean = {
            'cifar10': (0.4914, 0.4822, 0.4465),
            'cifar100': (0.5071, 0.4867, 0.4408),
            'tinyImagenet': (0.485, 0.456, 0.406),
            'VGG':(0.547, 0.460, 0.404),
            }

    std = {
            'cifar10': (0.2023, 0.1994, 0.2010),
            'cifar100': (0.2675, 0.2565, 0.2761),
            'tinyImagenet': (0.229, 0.224, 0.225),
            'VGG':[0.323, 0.298, 0.263]            
            }

    transform_dset_list = [
            transforms.ToTensor(),
            transforms.Normalize(mean[opt.dataset],std[opt.dataset]),
        ]

    transform_dset2 = transforms.Compose(
        [   transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean[opt.dataset],std[opt.dataset]),
        ]
    )
    if opt.model ==  'ViT':
        transform_dset_list.insert(2,transforms.Resize(224,antialias=True))
    transform_dset = transforms.Compose(transform_dset_list)
    
    if opt.dataset == 'cifar10':
        train_set = torchvision.datasets.CIFAR10(root=opt.data_path, train=True, download=True, transform=transform_dset)
        held_out = torchvision.datasets.CIFAR10(root=opt.data_path, train=False, download=True, transform=transform_dset)
        if file_fgt is None:
            forget_idx = np.loadtxt('./forget_idx_5000_cifar10.txt').astype(np.int64)
        else:
            forget_idx = np.loadtxt(file_fgt).astype(np.int64)



    elif opt.dataset=='cifar100':
        train_set = torchvision.datasets.CIFAR100(root=opt.data_path, train=True, download=True, transform=transform_dset)
        held_out = torchvision.datasets.CIFAR100(root=opt.data_path, train=False, download=True, transform=transform_dset)
        #use numpy modules to read txt file for cifar100
        if file_fgt is None:
            forget_idx = np.loadtxt('./forget_idx_5000_cifar100.txt').astype(np.int64)
        else:
            forget_idx = np.loadtxt(file_fgt).astype(np.int64)

    elif opt.dataset == 'tinyImagenet':
        train_set = torchvision.datasets.ImageFolder(root=opt.data_path+'/tiny-imagenet-200/train/',transform=transform_dset)
        held_out = torchvision.datasets.ImageFolder(root=opt.data_path+'/tiny-imagenet-200/val/images/',transform=transform_dset)
        if file_fgt is None:
            forget_idx = np.loadtxt('./forget_idx_5000_tinyImagenet.txt').astype(np.int64)
        else:
            forget_idx = np.loadtxt(file_fgt).astype(np.int64)
    
        
        
    
    train_loader = DataLoader(train_set, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)

    
    ### get held out dataset for generating test and validation 
    
    test_set, val_set = random_split(held_out, [0.5, 0.5])
    test_loader = DataLoader(test_set, batch_size=opt.batch_size, drop_last=True, shuffle=False, num_workers=opt.num_workers)
    val_loader = DataLoader(val_set, batch_size=opt.batch_size, drop_last=True, shuffle=False, num_workers=opt.num_workers)

    

    # construct indices of retain from those of the forget set
    forget_mask = np.zeros(len(train_set.targets), dtype=bool)
    forget_mask[forget_idx] = True
    retain_idx = np.arange(forget_mask.size)[~forget_mask]

    forget_set = Subset(train_set, forget_idx)
    retain_set = Subset(train_set, retain_idx)


    train_forget_loader = DataLoader(forget_set, batch_size=opt.batch_size, drop_last=True, shuffle=False, num_workers=opt.num_workers)
    train_retain_loader = DataLoader(retain_set, batch_size=opt.batch_size, drop_last=True, shuffle=True, num_workers=opt.num_workers)

    return train_loader, test_loader, train_forget_loader, train_retain_loader

class CustomDataset_10subj(Dataset):
    def __init__(self, df_all,path,best_10_subject,transform=None,train=True,split=False,retain=False,class_to_remove=[5,]):
        
        self.df_all = df_all
        self.transform = transform
        self.path = path
        N = self.df_all.shape[0]
        if train:
            self.df = df_all.iloc[:int(0.80*N),:]
        else:
            self.df = df_all.iloc[int(0.80*N):,:]

        list_remove = [best_10_subject[i] for i in class_to_remove]
        ### filter for retain and forget if necessary
        if split:
            if retain:
                mask = self.df.Id.apply(lambda x: any(item for item in list_remove if not(item in x)))
                self.df = self.df[mask]
                 
            else:
                mask = self.df.Id.apply(lambda x: any(item for item in list_remove if item in x))
                self.df = self.df[mask]
        else:
            pass
        self.best_10_subject = best_10_subject
        self.map_subj_to_class()
        self.img_paths = self.df.iloc[:, 0]
        self.targets = torch.tensor([self.dictionary_class[i.split('/')[0]] for i in self.df.iloc[:, 0].to_list()])



    def __len__(self):
        return len(self.df)
    # def transform_labels:
    def map_subj_to_class(self):
        self.dictionary_class = {}
        cnt=0
        for subj in self.best_10_subject:
            self.dictionary_class[subj] = cnt
            cnt+=1

    def __getitem__(self, idx):
        img_path = self.img_paths.iloc[idx]
        image = Image.open(self.path+img_path).convert('RGB')
        label = self.targets[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


def get_dsets_shadow(class_to_remove=None,file_fgt=None):


    mean = {
            'cifar10': (0.4914, 0.4822, 0.4465),
            'cifar100': (0.5071, 0.4867, 0.4408),
            'tinyImagenet': (0.485, 0.456, 0.406),
            'VGG':(0.547, 0.460, 0.404)
            }

    std = {
            'cifar10': (0.2023, 0.1994, 0.2010),
            'cifar100': (0.2675, 0.2565, 0.2761),
            'tinyImagenet': (0.229, 0.224, 0.225),
            'VGG':(0.323, 0.298, 0.263)
            }


    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean[opt.dataset], std=std[opt.dataset])
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean[opt.dataset], std=std[opt.dataset])
    ])

    transform_train_tiny = transforms.Compose([
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean[opt.dataset], std=std[opt.dataset])
        ])

    transform_test_tiny = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean[opt.dataset], std=std[opt.dataset])
        ])

    # we split held out - train
    if opt.dataset == 'cifar10':
        train_set = torchvision.datasets.CIFAR10(root=opt.data_path, train=True, download=True, transform=transform_train)
        test_set = torchvision.datasets.CIFAR10(root=opt.data_path, train=False, download=True, transform=transform_test)

    elif opt.dataset == 'cifar100':
        train_set = torchvision.datasets.CIFAR100(root=opt.data_path, train=True, download=True, transform=transform_train)
        test_set = torchvision.datasets.CIFAR100(root=opt.data_path, train=False, download=True, transform=transform_test)
        
    elif opt.dataset == 'tinyImagenet':
        train_set = torchvision.datasets.ImageFolder(root=opt.data_path+'/tiny-imagenet-200/train',transform=transform_train_tiny)
        test_set = torchvision.datasets.ImageFolder(root=opt.data_path+'/tiny-imagenet-200/val/images',transform=transform_test_tiny)

    if file_fgt is not None and class_to_remove is None:
        forget_idx = np.loadtxt(file_fgt).astype(np.int64)

        forget_mask = np.zeros(len(train_set.targets), dtype=bool)
        forget_mask[forget_idx] = True
        retain_idx = np.arange(forget_mask.size)[~forget_mask]

        forget_set = Subset(train_set, forget_idx)
        retain_set = Subset(train_set, retain_idx)
        return forget_set, retain_set, test_set

    elif file_fgt is None and class_to_remove is not None:    
        test_forget_set, test_retain_set = split_retain_forget(test_set, class_to_remove)
        forget_set, retain_set = split_retain_forget(train_set, class_to_remove)

        return forget_set, retain_set, test_forget_set, test_retain_set


def get_dsets_shadow(file_fgt=None):
    mean = {
            'cifar10': (0.4914, 0.4822, 0.4465),
            'cifar100': (0.5071, 0.4867, 0.4408),
            'tinyImagenet': (0.485, 0.456, 0.406),
            'VGG':(0.547, 0.460, 0.404),
            }

    std = {
            'cifar10': (0.2023, 0.1994, 0.2010),
            'cifar100': (0.2675, 0.2565, 0.2761),
            'tinyImagenet': (0.229, 0.224, 0.225),
            'VGG':[0.323, 0.298, 0.263]            
            }

    transform_dset_list = [
            transforms.ToTensor(),
            transforms.Normalize(mean[opt.dataset],std[opt.dataset]),
        ]

    transform_dset2 = transforms.Compose(
        [   transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean[opt.dataset],std[opt.dataset]),
        ]
    )
    if opt.model ==  'ViT':
        transform_dset_list.insert(2,transforms.Resize(224,antialias=True))
    transform_dset = transforms.Compose(transform_dset_list)
    
    if opt.dataset == 'cifar10':
        train_set = torchvision.datasets.CIFAR10(root=opt.data_path, train=True, download=True, transform=transform_dset)
        held_out = torchvision.datasets.CIFAR10(root=opt.data_path, train=False, download=True, transform=transform_dset)
        if file_fgt is None:
            forget_idx = np.loadtxt('./forget_idx_5000_cifar10.txt').astype(np.int64)
        else:
            forget_idx = np.loadtxt(file_fgt).astype(np.int64)



    elif opt.dataset=='cifar100':
        train_set = torchvision.datasets.CIFAR100(root=opt.data_path, train=True, download=True, transform=transform_dset)
        held_out = torchvision.datasets.CIFAR100(root=opt.data_path, train=False, download=True, transform=transform_dset)
        #use numpy modules to read txt file for cifar100
        if file_fgt is None:
            forget_idx = np.loadtxt('./forget_idx_5000_cifar100.txt').astype(np.int64)
        else:
            forget_idx = np.loadtxt(file_fgt).astype(np.int64)

    elif opt.dataset == 'tinyImagenet':
        train_set = torchvision.datasets.ImageFolder(root=opt.data_path+'/tiny-imagenet-200/train/',transform=transform_dset)
        held_out = torchvision.datasets.ImageFolder(root=opt.data_path+'/tiny-imagenet-200/val/images/',transform=transform_dset)
        if file_fgt is None:
            forget_idx = np.loadtxt('./forget_idx_5000_tinyImagenet.txt').astype(np.int64)
        else:
            forget_idx = np.loadtxt(file_fgt).astype(np.int64)
    
        
        
    
    train_loader = DataLoader(train_set, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)

    
    ### get held out dataset for generating test and validation 
    
    test_set, val_set = random_split(held_out, [0.5, 0.5])
    test_loader = DataLoader(test_set, batch_size=opt.batch_size, drop_last=True, shuffle=False, num_workers=opt.num_workers)
    val_loader = DataLoader(val_set, batch_size=opt.batch_size, drop_last=True, shuffle=False, num_workers=opt.num_workers)

    

    # construct indices of retain from those of the forget set
    forget_mask = np.zeros(len(train_set.targets), dtype=bool)
    forget_mask[forget_idx] = True
    retain_idx = np.arange(forget_mask.size)[~forget_mask]

    forget_set = Subset(train_set, forget_idx)
    retain_set = Subset(train_set, retain_idx)


    train_forget_loader = DataLoader(forget_set, batch_size=opt.batch_size, drop_last=True, shuffle=False, num_workers=opt.num_workers)
    train_retain_loader = DataLoader(retain_set, batch_size=opt.batch_size, drop_last=True, shuffle=True, num_workers=opt.num_workers)

    return train_loader, test_loader, train_forget_loader, train_retain_loader