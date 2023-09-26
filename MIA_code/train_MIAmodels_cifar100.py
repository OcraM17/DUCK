import torch
import torchvision
#from torchvision.models import resnet18,ResNet18_Weights,resnet50,ResNet50_Weights
from resnet import resnet18
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
import pickle as pk

from MIA_code.dsets_cifar100 import transform_train,transform_test,OPT,Custom_cifar100_Dataset
from MIA_code.utils import train_model, obtain_MIA_data,compute_accuracy


if __name__ == "__main__":  
    opt = OPT
    # Create the parser
    parser = argparse.ArgumentParser(description="This is a description of what this script does")

    # Add the arguments
    parser.add_argument('--run', type=int, default=1)
    parser.add_argument('--test', type=bool, default=False)
    # Execute the parse_args() method
    args = parser.parse_args()
    
    print(f'RUN num: {args.run}')

    ###############################################
    cifar100_training = Custom_cifar100_Dataset(root=opt.data, train=True,transform=transform_train,runs=args.run)
    training_loader = DataLoader(cifar100_training, shuffle=True, num_workers=opt.num_workers, batch_size=opt.batch_size)
    
    cifar100_test = Custom_cifar100_Dataset(root=opt.data, train=False, transform=transform_test,runs=args.run)
    test_loader = DataLoader(cifar100_test, shuffle=False, num_workers=opt.num_workers, batch_size=opt.batch_size)

    print(f'training dataloader batches: {len(training_loader)}')
    print(f'test dataloader batches: {len(test_loader)}')

    # Load the pretrained model
    model =resnet18()#weights=ResNet18_Weights.IMAGENET1K_V1)
    # Change the final layer
    #model.fc = nn.Linear(model.fc.in_features, 100)#nn.Sequential(nn.Dropout(p=0.1),)#0.2

    model = model.to(opt.device)

    # Define the criterion and the optimizer
    criterion = nn.CrossEntropyLoss()#label_smoothing=.06) #weight=class_weights)
    #optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr,weight_decay=opt.wd)
    # Use Cosine Annealing scheduler
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.epochs)
    #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60,120,160], gamma=0.2, last_epoch=-1, verbose=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 250], gamma=0.1)
    # Train the model
    opt.model_weights_name = opt.model_weights_name +f'_run_{args.run}'

    train_model(model,optimizer,criterion,scheduler,training_loader,test_loader,opt,args,plot_name=None)
    
    ##### re do the dataset and dataloader because of tranformations 
    cifar100_training = Custom_cifar100_Dataset(root=opt.data, train=True,transform=transform_test,runs=args.run)
    training_loader = torch.utils.data.DataLoader(cifar100_training, batch_size=opt.batch_size, shuffle=True,num_workers=opt.num_workers)
    #####
    classes,predictions,case = obtain_MIA_data(model,training_loader,opt,train=True)

    classes_test,predictions_test,case_test = obtain_MIA_data(model,test_loader,opt,train=False)

    classes = np.concatenate((classes,classes_test),axis=0)
    predictions = np.concatenate((predictions,predictions_test),axis=0)
    case = np.concatenate((case,case_test),axis=0)

    #save files
    filename = opt.outputs_from_model+'_MIA_params_run_'+str(args.run)+'.pkl'
    file = open(filename,'wb')
    pk.dump([classes,predictions,case],file)