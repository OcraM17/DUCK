
import os
import requests
import numpy as np
import sys
import torch

from copy import deepcopy
from dsets import get_dsets_remove_class, get_dsets
#to clean up
from utils import accuracy, compute_metrics,get_resnet18_trained_on_cifar10, set_seed, compute_losses, simple_mia, get_retrained_model, get_allcnn_trained_on_cifar10,get_resnet50_trained_on_VGGFace_10_subjects,get_resnet18_trained

from unlearn import unlearning
from MIA_code.MIA import get_MIA_MLP
from opts import OPT as opt
import pickle as pk
import torch.nn as nn
import pickle as pk
import matplotlib.pyplot as plt
from publisher import push_results

def main():
    # set random seed
    set_seed(opt.seed)
    
    #set all df to None
    df_or_model = None
    df_un_model = None
    df_rt_model = None 

    ##### GET DATA #####
    if opt.class_to_be_removed is None:
        train_loader, test_loader, train_fgt_loader, train_retain_loader = get_dsets()
    else:
        all_train_loader,all_test_loader, train_fgt_loader, train_retain_loader, test_fgt_loader, test_retain_loader = get_dsets_remove_class(opt.class_to_be_removed)

    ##### GET MODEL ##### 
    # function to be fixed

    if opt.model== 'resnet18':
        original_pretr_model = get_resnet18_trained()

    elif opt.model== 'resnet50' and opt.dataset == 'VGG':
        original_pretr_model = get_resnet50_trained_on_VGGFace_10_subjects()
    else:
        
        raise NotImplementedError

    
    original_pretr_model.to(opt.device)
    original_pretr_model.eval()
    if opt.run_original:
        print('\n----ORIGINAL MODEL----')
        if opt.class_to_be_removed is None:
            print(f"TEST-LOADER:{accuracy(original_pretr_model, test_loader):.3f} \nFORGET-LOADER: {accuracy(original_pretr_model, train_fgt_loader):.3f}\nRETAIN-LOADER: {accuracy(original_pretr_model, train_retain_loader):.3f}  ")
            #MIA
            df_or_model = get_MIA_MLP(train_fgt_loader, test_loader, original_pretr_model, opt)
        else:
            print('TRAIN:')
            print(f'FORGET-LOADER: {accuracy(original_pretr_model,train_fgt_loader ):.3f}\nRETAIN-LOADER: {accuracy(original_pretr_model, train_retain_loader):.3f}')
            print('TEST:')
            print(f'FORGET-LOADER: {accuracy(original_pretr_model, test_fgt_loader):.3f}\nRETAIN-LOADER: {accuracy(original_pretr_model,test_retain_loader ):.3f}')
            #MIA
            df_or_model = get_MIA_MLP(train_fgt_loader, test_fgt_loader, original_pretr_model, opt)
        #print(df_or_model)
        print('Results MIA:\n',df_or_model.mean(0))

    ##### UNLEARN #####
    pretr_model = deepcopy(original_pretr_model)
    pretr_model.fc = nn.Sequential(nn.Dropout(0.4),pretr_model.fc) 
    pretr_model.to(opt.device)
    pretr_model.eval()

    if opt.run_unlearn:
        print('\n----- UNLEARNED ----')
        unlearned_model = unlearning(pretr_model, train_retain_loader, train_fgt_loader,target_accuracy=opt.target_accuracy)

        if opt.class_to_be_removed is None:
            print(f"TEST-LOADER:{accuracy(unlearned_model, test_loader):.3f} \nFORGET-LOADER: {accuracy(unlearned_model, train_fgt_loader):.3f}\nRETAIN-LOADER: {accuracy(unlearned_model, train_retain_loader):.3f}  ")
            #MIA
            df_un_model = get_MIA_MLP(train_fgt_loader, test_loader, unlearned_model, opt)

        else:
            print('TRAIN:')
            print(f'FORGET-LOADER: {accuracy(unlearned_model,train_fgt_loader ):.3f}\nRETAIN-LOADER: {accuracy(unlearned_model, train_retain_loader):.3f}')
            print('TEST:')
            print(f'FORGET-LOADER: {accuracy(unlearned_model, test_fgt_loader):.3f}\nRETAIN-LOADER: {accuracy(unlearned_model,test_retain_loader ):.3f}')
            #MIA
            df_un_model = get_MIA_MLP(train_fgt_loader, test_fgt_loader, unlearned_model, opt)
        #print(df_un_model)
        print('Results MIA:\n',df_un_model.mean(0))

    if opt.run_rt_model:
        print('\n----RETRAINED on RETAIN ----')
        # RETRAINED MODEL ON RETAIN SET
        rt_model = get_retrained_model()
        rt_model.to(opt.device)
        rt_model.eval()
        if opt.class_to_be_removed is None:
            print(f"TEST-LOADER:{accuracy(rt_model, test_loader):.3f} \nFORGET-LOADER: {accuracy(rt_model, train_fgt_loader):.3f}\nRETAIN-LOADER: {accuracy(rt_model, train_retain_loader):.3f}  ")
            #MIA
            df_rt_model = get_MIA_MLP(train_fgt_loader, test_loader, rt_model, opt)

        else:
            print('TRAIN:')
            print(f'FORGET-LOADER: {accuracy(rt_model,train_fgt_loader ):.3f}\nRETAIN-LOADER: {accuracy(rt_model, train_retain_loader):.3f}')
            print('TEST:')
            print(f'FORGET-LOADER: {accuracy(rt_model, test_fgt_loader):.3f}\nRETAIN-LOADER: {accuracy(rt_model,test_retain_loader ):.3f}')
            #MIA
            df_rt_model = get_MIA_MLP(train_fgt_loader, test_fgt_loader, rt_model, opt)
        #print(df_un_model)
        print('Results MIA:\n',df_rt_model.mean(0))

    push_results(opt, df_or_model, df_un_model, df_rt_model)

if __name__ == "__main__":
    main()
