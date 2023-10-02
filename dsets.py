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
    if type(class_to_remove) is tuple:
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
            'tinyImagenet': (0.485, 0.456, 0.406)
            }

    std = {
            'cifar10': (0.2023, 0.1994, 0.2010),
            'cifar100': (0.2675, 0.2565, 0.2761),
            'tinyImagenet': (0.229, 0.224, 0.225)
            }

    # download and pre-process CIFAR10
    transform_dset = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean[opt.dataset],std[opt.dataset]),
        ]
    )

    # we split held out - train
    if opt.dataset == 'cifar10':
        train_set = torchvision.datasets.CIFAR10(root=opt.data_path, train=True, download=True, transform=transform_dset)
        test_set = torchvision.datasets.CIFAR10(root=opt.data_path, train=False, download=True, transform=transform_dset)
    elif opt.dataset == 'cifar100':
        train_set = torchvision.datasets.CIFAR100(root=opt.data_path, train=True, download=True, transform=transform_dset)
        test_set = torchvision.datasets.CIFAR100(root=opt.data_path, train=False, download=True, transform=transform_dset)
    elif opt.dataset == 'tinyImagenet':
        train_set = torchvision.datasets.ImageFolder(root=opt.data_path+'/tiny-imagenet-200/train/',transform=transform_dset)
        test_set = torchvision.datasets.ImageFolder(root=opt.data_path+'/tiny-imagenet-200/val/images/',transform=transform_dset)
    #val_set, test_set = torch.utils.data.random_split(held_out, [0.7, 0.3])

    test_forget_set, test_retain_set = split_retain_forget(test_set, class_to_remove)
    forget_set, retain_set = split_retain_forget(train_set, class_to_remove)

    # validation set and its subsets 
    all_test_loader = DataLoader(test_set, batch_size=opt.batch_size, shuffle=False, num_workers=2)
    test_fgt_loader = DataLoader(test_forget_set, batch_size=opt.batch_size, shuffle=False, num_workers=2)
    test_retain_loader = DataLoader(test_retain_set, batch_size=opt.batch_size, shuffle=False, num_workers=2)
    
    # all train and its subsets
    all_train_loader = DataLoader(train_set, batch_size=opt.batch_size, shuffle=False, num_workers=2)
    train_fgt_loader = DataLoader(forget_set, batch_size=opt.batch_size, shuffle=False, num_workers=2)
    train_retain_loader = DataLoader(retain_set, batch_size=opt.batch_size, shuffle=False, num_workers=2)


    return all_train_loader,all_test_loader, train_fgt_loader, train_retain_loader, test_fgt_loader, test_retain_loader




def get_dsets():
    mean = {
            'cifar10': (0.4914, 0.4822, 0.4465),
            'cifar100': (0.5071, 0.4867, 0.4408),
            'tinyImagenet': (0.485, 0.456, 0.406)
            }

    std = {
            'cifar10': (0.2023, 0.1994, 0.2010),
            'cifar100': (0.2675, 0.2565, 0.2761),
            'tinyImagenet': (0.229, 0.224, 0.225)
            }

    transform_dset = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean[opt.dataset],std[opt.dataset]),
        ]
    )
    
    if opt.dataset == 'cifar10':
        train_set = torchvision.datasets.CIFAR10(root=opt.data_path, train=True, download=True, transform=transform_dset)
        held_out = torchvision.datasets.CIFAR10(root=opt.data_path, train=False, download=True, transform=transform_dset)
        forget_idx = np.loadtxt('./forget_idx_5000_cifar10.txt').astype(np.int64)


    elif opt.dataset=='cifar100':
        train_set = torchvision.datasets.CIFAR100(root=opt.data_path, train=True, download=True, transform=transform_dset)
        held_out = torchvision.datasets.CIFAR100(root=opt.data_path, train=False, download=True, transform=transform_dset)
        #use numpy modules to read txt file for cifar100
        forget_idx = np.loadtxt('./forget_idx_5000_cifar100.txt').astype(np.int64)

    elif opt.dataset == 'tinyImagenet':
        train_set = torchvision.datasets.ImageFolder(root=opt.data_path+'/tiny-imagenet-200/train/',transform=transform_dset)
        held_out = torchvision.datasets.ImageFolder(root=opt.data_path+'/tiny-imagenet-200/val/images/',transform=transform_dset)
        forget_idx = np.loadtxt('./forget_idx_5000_tinyImagenet.txt').astype(np.int64)
        
    else:
        raise NotImplementedError
    
    train_loader = DataLoader(train_set, batch_size=opt.batch_size, shuffle=False, num_workers=2)

    
    ### get held out dataset for generating test and validation 
    
    test_set, val_set = random_split(held_out, [0.5, 0.5])
    test_loader = DataLoader(test_set, batch_size=opt.batch_size, drop_last=True, shuffle=False, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=opt.batch_size, drop_last=True, shuffle=False, num_workers=2)

    

    # construct indices of retain from those of the forget set
    forget_mask = np.zeros(len(train_set.targets), dtype=bool)
    forget_mask[forget_idx] = True
    retain_idx = np.arange(forget_mask.size)[~forget_mask]

    forget_set = Subset(train_set, forget_idx)
    retain_set = Subset(train_set, retain_idx)


    train_forget_loader = DataLoader(forget_set, batch_size=opt.batch_size, drop_last=True, shuffle=False, num_workers=4)
    train_retain_loader = DataLoader(retain_set, batch_size=opt.batch_size, drop_last=True, shuffle=True, num_workers=4)

    return train_loader, test_loader, train_forget_loader, train_retain_loader



##############################################################################################################
##############################################################################################################
##############################################################################################################

class CustomDataset(Dataset):
    def __init__(self, df_all,path, transform=None,train=True,split=False,retain=False):
        
        self.df_all = df_all
        self.transform = transform
        self.path = path
        N = self.df_all.shape[0]
        if train and not(split):
            self.df = df_all.iloc[:int(0.75*N),:]
        elif train and retain and split:
            self.df_buff = df_all.iloc[:int(0.75*N),:]
            self.subject = self.split_subjects(retain)
            print(f'num of subjects retain: {len(self.subject)}')
            mask = self.df_buff.Id.apply(lambda x: any(item for item in self.subject if item in x))
            self.df = self.df_buff[mask]

        elif train and not(retain) and split:
            self.df_buff = df_all.iloc[:int(0.75*N),:]
            self.subject = self.split_subjects(retain)
            print(f'num of subjects forget: {len(self.subject)}')

            mask = self.df_buff.Id.apply(lambda x: any(item for item in self.subject if item in x))
            self.df = self.df_buff[mask]
            #print(self.df.head(20))
        else:
            self.df = df_all.iloc[int(0.75*N):,:]
    
    def __len__(self):
        return len(self.df)
    # def transform_labels:
    
    def split_subjects(self,retain):
        #select subjects for forget set with avg age <=20
        subjects = []
        subjects_fgt = []
        mean_age = []
        cnt=0
        for i in range(self.df_buff.shape[0]):
            name = self.df_buff.iloc[i,0].split('/')[0]
            age = self.df_buff.iloc[i,3]

            if not(name in subjects):
                if not(retain) and cnt<4 and age<=20:#fgt
                    subjects_fgt.append(name)                
                    cnt+=1
                subjects.append(name)

        if retain:
            subjects_retain = [i for i in subjects if i not in subjects_fgt]
            return subjects
        else:
            return subjects_fgt
        
    def __getitem__(self, idx):
        img_path = self.df.iloc[idx, 0]
        image = Image.open(self.path+img_path).convert('RGB')
        label = self.df.iloc[idx, -1]

        if self.transform:
            image = self.transform(image)

        return image, label


class OPT_VGG:
    # Data
    PATH = '/home/jb/data/VGG-Face2/data/train/'
    data_path = '/home/jb/data/VGG-Face2/train.age_detected_filtered.csv'
    num_workers = 4


def get_dsets_VGGFace():
    opt_vgg = OPT_VGG
    df = pd.read_csv(opt_vgg.data_path,sep=',',header=(0))
    numbers = { 
        0: [1688,0,10],
        1: [14622,10,20],
        2: [1688,20,30],
        3: [1130, 30,40],
        4: [643,40,48],
        5: [932,48,56],
        6: [2333,56,64],
        7: [1553,64,72],
        8: [689,72,80]
    }

    #add the correct label
    class_weights = []

    for i in range(9):
        df_mod = df[((df['Age']>=numbers[i][1]) & (df['Age']<numbers[i][2]))].copy()
        df_mod['labels'] = i
        class_weights.append(df.shape[0]/df_mod.shape[0]*9)    
        if i ==0:
            df_out = df_mod
        else:
            df_out = pd.concat([df_out,df_mod],axis=0)


    #shuffle data
    df_out = df_out.sample(frac = 1)
        # download and pre-process CIFAR10

    # Load and transform the data
    transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.RandomCrop(128),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.RandomChoice([
    transforms.ColorJitter(brightness=.5,hue=.3),
    transforms.GaussianBlur(kernel_size=5,sigma=(1,2.5)),
    transforms.RandomGrayscale(p=.8)]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.547, 0.460, 0.404], std=[0.323, 0.298, 0.263]),
    ])

    transform_test = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.547, 0.460, 0.404], std=[0.323, 0.298, 0.263]),
    ])
    
    # we split held out data into test and validation set
    trainset = CustomDataset(df_out,path = opt_vgg.PATH, train= True, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True,num_workers=opt.num_workers)

    testset = CustomDataset(df_out,path = opt_vgg.PATH, train=False, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False,num_workers=opt.num_workers)
    
    retain_set = CustomDataset(df_out,path = opt_vgg.PATH, train= True,split=True,retain=True, transform=transform_test)
    forget_set = CustomDataset(df_out,path = opt_vgg.PATH, train= True,split=True,retain=False, transform=transform_test)

    forget_loader = DataLoader(forget_set, batch_size=opt.batch_size, drop_last=True, shuffle=False, num_workers=4)
    retain_loader = DataLoader(retain_set, batch_size=opt.batch_size, drop_last=True, shuffle=True, num_workers=4)
    retain_loader2 = DataLoader(retain_set, batch_size=opt.batch_size_FT, drop_last=True, shuffle=True, num_workers=4)


    return train_loader, test_loader, forget_loader, retain_loader, retain_loader2

########################## paper exp VGG #######################################
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
        img_path = self.df.iloc[idx, 0]
        image = Image.open(self.path+img_path).convert('RGB')
        label = self.dictionary_class[self.df.iloc[idx, 0].split('/')[0]]

        if self.transform:
            image = self.transform(image)

        return image, label


class OPT_VGG_10_subjects:
    # Data
    PATH = '/home/jb/data/VGG-Face2/data/train/'
    data_path = '/home/jb/data/VGG-Face2/data/train_list.txt'
    num_workers = 4


def get_dsets_VGGFace_10_subjects():
    opt_vgg = OPT_VGG_10_subjects
    np.random.seed(42)

    folder_list = glob.glob(opt_vgg.PATH+'*')

    dict_subj = {}
    for fold in folder_list:
        files = glob.glob(fold+'/*.jpg')
        
        if len(files)>500:
            dict_subj[fold.split('/')[-1]] = len(files)
    print(f'Num subject suitable: {len(list(dict_subj.keys()))}')


    df = pd.read_csv(opt_vgg.data_path,sep=',',header=None, names=['Id',])#pd.read_csv(opt.data_path,sep=',',header=(0))


    sorted_dict_subj = sorted(dict_subj.items(), key=lambda x:x[1], reverse=True)
    sorted_dict_subj = dict(sorted_dict_subj)

    best_10_subject=[]
    skip = list(sorted_dict_subj.keys())[9]
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

    # Load and transform the data
    transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.RandomCrop(128),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.RandomChoice([
    transforms.ColorJitter(brightness=.5,hue=.3),
    transforms.GaussianBlur(kernel_size=5,sigma=(1,2.5)),
    transforms.RandomGrayscale(p=.8)]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.547, 0.460, 0.404], std=[0.323, 0.298, 0.263]),
    ])

    transform_test = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.547, 0.460, 0.404], std=[0.323, 0.298, 0.263]),
    ])
    
    # we split held out data into test and validation set
    trainset = CustomDataset_10subj(df,path = opt_vgg.PATH,best_10_subject=best_10_subject, train= True, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True,num_workers=opt.num_workers)

    testset = CustomDataset_10subj(df,path = opt_vgg.PATH,best_10_subject=best_10_subject, train=False, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False,num_workers=opt.num_workers)
    
    retain_set = CustomDataset_10subj(df,path = opt_vgg.PATH,best_10_subject=best_10_subject, train= True,split=True,retain=True, transform=transform_test,class_to_remove=[opt.class_to_be_removed,])
    forget_set = CustomDataset_10subj(df,path = opt_vgg.PATH,best_10_subject=best_10_subject, train= True,split=True,retain=False, transform=transform_test,class_to_remove=[opt.class_to_be_removed,])

    forget_loader = DataLoader(forget_set, batch_size=opt.batch_size, drop_last=True, shuffle=False, num_workers=4)
    retain_loader = DataLoader(retain_set, batch_size=opt.batch_size, drop_last=True, shuffle=True, num_workers=4)
    

    testset_retain = CustomDataset_10subj(df,path = opt_vgg.PATH,best_10_subject=best_10_subject, train=False,split=True,retain=True,transform=transform_test,class_to_remove=[opt.class_to_be_removed,])
    test_loader_retain = torch.utils.data.DataLoader(testset_retain, batch_size=opt.batch_size, shuffle=False,num_workers=opt.num_workers)

    testset_forget = CustomDataset_10subj(df,path = opt_vgg.PATH,best_10_subject=best_10_subject, train=False,split=True,retain=False,transform=transform_test,class_to_remove=[opt.class_to_be_removed,])
    test_loader_forget = torch.utils.data.DataLoader(testset_forget, batch_size=opt.batch_size, shuffle=False,num_workers=opt.num_workers)

    return train_loader, test_loader, forget_loader, retain_loader, test_loader_retain,test_loader_forget