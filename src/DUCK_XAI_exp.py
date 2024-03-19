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

class_to_remove = 7
seed=42
class_names = classes = ['plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck']

import torchvision
from torchvision.models.resnet import resnet18,ResNet18_Weights

original_pretr_model = get_trained_model()

un_model = deepcopy(original_pretr_model)
original_pretr_model.to(opt.device)
original_pretr_model.eval()

#load weights
unlearned_model_dict = torch.load(f"{opt.root_folder}/out/{opt.mode}/{opt.dataset}/models/unlearned_model_{opt.method}_seed_{seed}_class_{class_to_remove}.pth")
un_model.load_state_dict(unlearned_model_dict)
un_model_s = nn.Sequential(nn.Softmax(), un_model)
un_model_s.eval()
un_model_s.to(opt.device)

opt.RT_model_weights_path = f'/home/jb/Documents/MachineUnlearning/src/weights/chks_cifar10/best_checkpoint_without_{class_to_remove}.pth'
rt_model = get_retrained_model()
rt_model.to(opt.device)
rt_model.eval()
#load data
_, _, train_fgt_loader, train_retain_loader, test_fgt_loader, test_retain_loader = get_dsets_remove_class(class_to_remove)

# ### SHAP


x,y=next(iter(train_fgt_loader))
print('check',x.shape, torch.unique(y))
X = x.to(opt.device)
X = torch.permute(X,(0,2,3,1))
def predict(img) -> torch.Tensor:
    print(img.shape)
    if not(torch.is_tensor(img)):
        img = torch.tensor(img)
    img = torch.permute(img,(0,3,1,2))
    img = img.to(opt.device)
    output = rt_model(img)#original_pretr_model(img)#un_model_s(img)
    return output


print(X.shape)
out = predict(X[:10])
classes = torch.argmax(out, axis=1).cpu().numpy()
print(f"Classes: {classes}: {np.array(class_names)[classes]}")

topk = 10
batch_size = 50
n_evals = 20000

# define a masker that is used to mask out partitions of the input image.
masker_blur = shap.maskers.Image("blur(16,16)", X[0].shape)

# create an explainer with model and image masker
explainer = shap.Explainer(predict, masker_blur, output_names=class_names)

# feed only one image
# here we explain two images using 100 evaluations of the underlying model to estimate the SHAP values
shap_values = explainer(
    X[15:16],#20 plane #15 horse
    max_evals=n_evals,
    batch_size=batch_size,
    outputs=shap.Explanation.argsort.flip[:topk],
)
print(shap_values.data.shape, shap_values.values.shape)

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
plt.savefig(f'duck_xai_class_fgt{class_to_remove}_ret.svg')
