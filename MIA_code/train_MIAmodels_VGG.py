import torch
import torchvision
from torchvision.models import resnet18,ResNet18_Weights,resnet50,ResNet50_Weights
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

from MIA_code.dsets_VGG import transform,transform_test,CustomDataset_10subj,OPT
from MIA_code.utils import train_model, obtain_MIA_data,compute_accuracy

if __name__ == "__main__":  
    # Create the parser
    parser = argparse.ArgumentParser(description="This is a description of what this script does")

    # Add the arguments
    parser.add_argument('--run', type=int, default=1)
    parser.add_argument('--test', type=bool, default=False)
    # Execute the parse_args() method
    args = parser.parse_args()
    
    print(f'RUN num: {args.run}')

    #################################### prepare data

    np.random.seed(42)
    opt = OPT
    folder_list = glob.glob(opt.PATH+'*')

    dict_subj = {}
    for fold in folder_list:
        files = glob.glob(fold+'/*.jpg')
        
        if len(files)>500:
            dict_subj[fold.split('/')[-1]] = len(files)
    print(f'Num subject suitable: {len(list(dict_subj.keys()))}')


    df = pd.read_csv(opt.data_path,sep=',',header=None, names=['Id',])#pd.read_csv(opt.data_path,sep=',',header=(0))

    sorted_dict_subj = sorted(dict_subj.items(), key=lambda x:x[1], reverse=True)
    sorted_dict_subj = dict(sorted_dict_subj)

    best_10_subject=[]
    skip = list(sorted_dict_subj.keys())[9]
    for key in sorted_dict_subj.keys():
        if key!=skip:
            best_10_subject.append(key)
            if len(best_10_subject)==10:
                break
    print(skip)
    print(best_10_subject)
    #filter for subjects
    mask = df.Id.apply(lambda x: any(item for item in best_10_subject if item in x))
    df = df[mask]

    #shuffle dataframe
    for i in range(args.run):
        df = df.sample(frac=1)

    ###############################################

    trainset = CustomDataset_10subj(df,path = opt.PATH,best_10_subject=best_10_subject, train= True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True,num_workers=opt.num_workers)

    testset = CustomDataset_10subj(df,path = opt.PATH, train=False,best_10_subject=best_10_subject, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False,num_workers=opt.num_workers)

    print(f'training dataloader batches: {len(trainloader)}')
    print(f'test dataloader batches: {len(testloader)}')

    # Load the pretrained model
    model =resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    # Change the final layer
    model.fc = nn.Sequential(nn.Dropout(p=0.2),nn.Linear(model.fc.in_features, 10))#0.2

    model = model.to(opt.device)
    # Define the criterion and the optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=.1) #weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr,weight_decay=opt.wd)
    # Use Cosine Annealing scheduler
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.epochs)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[7,12,18], gamma=0.5, last_epoch=-1, verbose=False)
    # Train the model
    opt.model_weights_name = opt.model_weights_name +f'_run_{args.run}'

    train_model(model,optimizer,criterion,scheduler,trainloader,testloader,opt,args,plot_name=None)
    
    ##### re do the dataset and dataloader because of tranformations 
    trainset = CustomDataset_10subj(df,path = opt.PATH,best_10_subject=best_10_subject, train= True, transform=transform_test)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True,num_workers=opt.num_workers)
    #####
    classes,predictions,case = obtain_MIA_data(model,trainloader,opt,train=True)

    classes_test,predictions_test,case_test = obtain_MIA_data(model,testloader,opt,train=False)

    classes = np.concatenate((classes,classes_test),axis=0)
    predictions = np.concatenate((predictions,predictions_test),axis=0)
    case = np.concatenate((case,case_test),axis=0)

    #save files
    filename = opt.outputs_from_model+'_MIA_params_run_'+str(args.run)+'.pkl'
    file = open(filename,'wb')
    pk.dump([classes,predictions,case],file)