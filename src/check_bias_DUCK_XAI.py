from copy import deepcopy
from dsets import get_dsets_remove_class, get_dsets
import pandas as pd

from utils import accuracy, set_seed, get_retrained_model,get_trained_model

from opts import OPT as opt
import torch.nn as nn
from tqdm import tqdm
import os
import torch

import json
import shap
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import numpy as np
from utils import set_seed
from torch import nn 
from torch.utils.data import DataLoader, Subset
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
        if self.get_id:
            return img, label, id
        else:
            return img, label
        

class_to_remove = 7
seed=42
class_names = classes = ['plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck']

import torchvision
from torchvision.models.resnet import resnet18,ResNet18_Weights

original_pretr_model = get_trained_model()
weights_path = '/home/jb/Documents/MachineUnlearning/poisoned_model2.pth'
original_pretr_model.load_state_dict(torch.load(weights_path))
un_model = deepcopy(original_pretr_model)
original_pretr_model.to(opt.device)
original_pretr_model.eval()

#load weights
unlearned_model_dict = torch.load('/home/jb/Documents/MachineUnlearning/model_xai_test.pth') #torch.load(f"{opt.root_folder}/out/{opt.mode}/{opt.dataset}/models/unlearned_model_{opt.method}_seed_{seed}_class_{class_to_remove}.pth")
un_model.load_state_dict(unlearned_model_dict)
un_model.eval()
un_model.to(opt.device)


opt.RT_model_weights_path = f'/home/jb/Documents/MachineUnlearning/poisoned_model2_retr.pth'
rt_model = deepcopy(original_pretr_model)
rt_model.load_state_dict(torch.load(opt.RT_model_weights_path))
rt_model.to(opt.device)
rt_model.eval()

#load data
transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                          torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
dataset=Poisoned_CIFAR10(transform,get_id=True)
dataloader=torch.utils.data.DataLoader(dataset, batch_size=2024, shuffle=True)

idxs = dataset.poisoned_indexes_car.numpy().astype(np.int64)#np.load('/home/jb/Documents/MachineUnlearning/poison/poisoned_indexes.npy').astype(np.int64)[:200]
forget_mask = np.zeros(len(dataset.imgs), dtype=bool)
forget_mask[idxs] = True
retain_idx = np.arange(forget_mask.size)[~forget_mask]
fgt_idx =np.arange(forget_mask.size)[forget_mask]
forget_set = Subset(dataset, fgt_idx)
retain_set = Subset(dataset, retain_idx)
fgt_loader=torch.utils.data.DataLoader(forget_set, batch_size=2024, shuffle=False)

# ### SHAP


x,y,id=next(iter(fgt_loader))
x = x[id==False]
y = y[id==False]
#x = x[(y==1) & (id==False)]
print(y)
print('check',x.shape, torch.unique(y))
X = x.to(opt.device)
X = torch.permute(X,(0,2,3,1))
def predict(img) -> torch.Tensor:
    print(img.shape)
    if not(torch.is_tensor(img)):
        img = torch.tensor(img)
    img = torch.permute(img,(0,3,1,2))
    img = img.to(opt.device)
    output = rt_model(img)###original_pretr_model(img)#original_pretr_model(img)#
    return output


print(X.shape)
out = predict(X[:10])
classes = torch.argmax(out, axis=1).cpu().numpy()
print('wwwww',out)
print(f"Classes: {classes}")

topk = 10
batch_size = 1000
n_evals = 20000
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

def standardize(img,mean,std):
    for i in range(3):
        img[:,:,:,i] = (img[:,:,:,i]*std[i])+mean[i]
    return img

# define a masker that is used to mask out partitions of the input image.
masker_blur = shap.maskers.Image("blur(16,16)", X[0].shape)
# create an explainer with model and image masker
explainer = shap.Explainer(predict, masker_blur, output_names=class_names)

# feed only one image
# here we explain two images using 100 evaluations of the underlying model to estimate the SHAP values

shap_values = explainer(
    X[2:3],#20 plane #15 horse
    max_evals=n_evals,
    batch_size=batch_size,
    outputs=shap.Explanation.argsort.flip[:topk],
)
print(shap_values.data.shape, shap_values.values.shape)

shap_values.data = standardize(shap_values.data,mean[opt.dataset],std[opt.dataset])

shap_values.data = (shap_values.data).cpu().numpy()[0]#inv_transform
shap_values.values = [val for val in np.moveaxis(shap_values.values[0], -1, 0)]


shap.image_plot(
    shap_values=shap_values.values,
    pixel_values=shap_values.data,
    labels=shap_values.output_names,
    true_labels=[class_names[class_to_remove]],
)
#plt.savefig(f'duck_xai_class_fgt{class_to_remove}.svg')
#plt.savefig(f'duck_xai_class_fgt{class_to_remove}_or.svg')
plt.savefig(f'duck_xai_bias_rt_model.svg')
#plt.savefig(f'duck_xai_bias_or.svg')

# fgt_loader=torch.utils.data.DataLoader(forget_set, batch_size=1, shuffle=False)
# mean_shap=[]
# for i,batch in enumerate(fgt_loader):
#     x = batch[0].to(opt.device)
#     x = torch.permute(x,(0,2,3,1))
#     shap_values = explainer(
#     x,#20 plane #15 horse
#     max_evals=n_evals,
#     batch_size=batch_size,
#     outputs=shap.Explanation.argsort.flip[:topk])
#     val = shap_values.values[:,0:4,28:32,:,0]
#     mean_shap.append(val.mean())
    
#     print(i,np.asarray(mean_shap).mean(),np.asarray(mean_shap).std())
        


# print(np.asarray(mean_shap).mean(),np.asarray(mean_shap).std())

# #numpy save array
# np.save('mean_shap_un.npy',np.asarray(mean_shap))