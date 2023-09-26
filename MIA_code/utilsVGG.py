import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, Dataset
import pandas as pd
from PIL import Image

from sklearn.metrics import confusion_matrix

import numpy as np



class OPT:

    seed = 42
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    
    # Data
    PATH = '/home/jb/data/VGG-Face2/data/train/'
    data_path = '/home/jb/data/VGG-Face2/data/train_list.txt'
    num_workers = 4
    batch_size = 256
    epochs = 20
    
    lr = 0.00005
    wd = 5e-4
    momentum = 0.9
    model_weights_name = '/home/jb/Documents/MachineUnlearning/weights/net_weights_resnet50_VGG_10sub'
    outputs_from_model = '/home/jb/Documents/MachineUnlearning/MIA_data/net_weights_resnet50_VGG_10sub'


class CustomDataset_10subj(Dataset):
    def __init__(self, df_all,path,best_10_subject,transform=None,train=True,):
        
        self.df_all = df_all
        self.transform = transform
        self.path = path
        N = self.df_all.shape[0]
        if train:
            self.df = df_all.iloc[:int(0.80*N),:]
        else:
            self.df = df_all.iloc[int(0.80*N):,:]
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