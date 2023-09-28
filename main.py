
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


def main():
    # set random seed
    set_seed(opt.seed)

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
     
    print('\n----ORIGINAL MODEL----')
    if opt.class_to_be_removed is None:
        print(f"TEST-LOADER:{accuracy(original_pretr_model, test_loader):.3f} \nFORGET-LOADER: {accuracy(original_pretr_model, train_fgt_loader):.3f}\nRETAIN-LOADER: {accuracy(original_pretr_model, train_retain_loader):.3f}  ")
        #MIA
        get_MIA_MLP(train_loader, test_loader, original_pretr_model, opt)
    else:
        print('TRAIN:')
        print(f'FORGET-LOADER: {accuracy(original_pretr_model,train_fgt_loader ):.3f}\nRETAIN-LOADER: {accuracy(original_pretr_model, train_retain_loader):.3f}')
        print('TEST:')
        print(f'FORGET-LOADER: {accuracy(original_pretr_model, test_fgt_loader):.3f}\nRETAIN-LOADER: {accuracy(original_pretr_model,test_retain_loader ):.3f}')
        #MIA
        get_MIA_MLP(train_fgt_loader, test_fgt_loader, original_pretr_model, opt)
   
        # ft_train_losses = compute_losses(original_pretr_model, train_retain_loader)
        # ft_forget_losses = compute_losses(original_pretr_model, train_fgt_loader)
        # ft_test_losses = compute_losses(original_pretr_model, test_retain_loader)
 
        # ft_forget_losses= ft_forget_losses[:len(ft_test_losses)]

        # ft_test_losses = ft_test_losses[:len(ft_forget_losses)]

        # ft_samples_mia = np.concatenate((ft_test_losses, ft_forget_losses)).reshape((-1, 1))
        # labels_mia = [0] * len(ft_test_losses) + [1] * len(ft_forget_losses)
        # ft_mia_scores = simple_mia(ft_samples_mia, labels_mia)
        # print(f"The MIA has an accuracy of {ft_mia_scores.mean():.3f} on forgotten vs unseen images")
        # input('')
    #get_outputs(retain_loader,forget_loader,original_pretr_model,'/home/jb/Documents/MachineUnlearning/res_original_model.pkl',opt=opt)


    ##### UNLEARN #####
    pretr_model = deepcopy(original_pretr_model)
    pretr_model.to(opt.device)
    pretr_model.eval()

    
    print('\n----- UNLEARNED ----')
    unlearned_model = unlearning(pretr_model, train_retain_loader, train_fgt_loader,target_accuracy=opt.target_accuracy)

    if opt.class_to_be_removed is None:
        print(f"TEST-LOADER:{accuracy(unlearned_model, test_loader):.3f} \nFORGET-LOADER: {accuracy(unlearned_model, train_fgt_loader):.3f}\nRETAIN-LOADER: {accuracy(unlearned_model, train_retain_loader):.3f}  ")
        #MIA
        get_MIA_MLP(train_fgt_loader, test_fgt_loader, unlearned_model, opt)

    else:
        print('TRAIN:')
        print(f'FORGET-LOADER: {accuracy(unlearned_model,train_fgt_loader ):.3f}\nRETAIN-LOADER: {accuracy(unlearned_model, train_retain_loader):.3f}')
        print('TEST:')
        print(f'FORGET-LOADER: {accuracy(unlearned_model, test_fgt_loader):.3f}\nRETAIN-LOADER: {accuracy(unlearned_model,test_retain_loader ):.3f}')
        #MIA
        get_MIA_MLP(train_fgt_loader, test_fgt_loader, unlearned_model, opt)




    # ft_forget_losses = compute_losses(unlearned_model, forget_loader)
    # ft_test_losses = compute_losses(unlearned_model, test_loader)

    # ft_forget_losses= ft_forget_losses[:len(ft_test_losses)]
    # ft_test_losses = ft_test_losses[:len(ft_forget_losses)]

    # ft_samples_mia = np.concatenate((ft_test_losses, ft_forget_losses)).reshape((-1, 1))
    # labels_mia = [0] * len(ft_test_losses) + [1] * len(ft_forget_losses)
    # ft_mia_scores = simple_mia(ft_samples_mia, labels_mia)
    # print(f"The MIA has an accuracy of {ft_mia_scores.mean():.3f} on forgotten vs unseen images")
    #get_outputs(retain_loader,forget_loader,unlearned_model,'/home/jb/Documents/MachineUnlearning/res_unlr_model.pkl',opt=opt)
    
    #compute_metrics(unlearned_model, train_loader, forget_loader, retain_loader, all_val_loader, val_fgt_loader, val_retain_loader)

    # print('\n-----FT on RETAIN----')

    #unlearned_model_finetuned = fine_tune2(unlearned_model, retain_loader2)
    #print(f"TEST-LOADER:{accuracy(unlearned_model_finetuned, test_loader):.3f} \nFORGET-LOADER: {accuracy(unlearned_model_finetuned, forget_loader):.3f}\nRETAIN-LOADER: {accuracy(unlearned_model_finetuned, retain_loader):.3f}  ")
    #ft_forget_losses = compute_losses(unlearned_model_finetuned, forget_loader)
    #ft_test_losses = compute_losses(unlearned_model_finetuned, test_loader)
    #ft_samples_mia = np.concatenate((ft_test_losses, ft_forget_losses)).reshape((-1, 1))
    #labels_mia = [0] * len(ft_test_losses) + [1] * len(ft_forget_losses)
    #ft_mia_scores = simple_mia(ft_samples_mia, labels_mia)
    #print(f"The MIA has an accuracy of {ft_mia_scores.mean():.3f} on forgotten vs unseen images")
    #compute_metrics(unlearned_model_finetuned, train_loader, forget_loader, retain_loader, all_val_loader, val_fgt_loader, val_retain_loader)



    #print('\n----RETRAINED on RETAIN ----')
    # RETRAINED MODEL ON RETAIN SET
    # WE SHOULD RETRAIN FROM SCRATCH 

    #rt_model = get_retrained_model(test_retain_loader, test_fgt_loader)
    #print(f"TEST-LOADER:{accuracy(rt_model, test_loader):.3f} \nFORGET-LOADER: {accuracy(rt_model, forget_loader):.3f}\nRETAIN-LOADER: {accuracy(rt_model, retain_loader):.3f}  ")
    #for _ in range(5):
    #    print(get_MIA_MLP(train_fgt_loader, test_fgt_loader, rt_model))
    # retrained_forget_losses = compute_losses(rt_model, forget_loader)
    # retrained_test_losses = compute_losses(rt_model, test_loader)

    # retrained_forget_losses= retrained_forget_losses[:len(retrained_test_losses)]
    # retrained_test_losses = retrained_test_losses[:len(retrained_forget_losses)]

    # retrained_samples_mia = np.concatenate((retrained_test_losses, retrained_forget_losses)).reshape((-1, 1))
    # labels_mia = [0] * len(retrained_test_losses) + [1] * len(retrained_forget_losses)
    # retrained_mia_scores = simple_mia(retrained_samples_mia, labels_mia)
    # print(f"MIA retrained {retrained_mia_scores.mean():.3f}")


    # retrained_samples_mia = np.concatenate((retrained_test_losses, ft_forget_losses)).reshape((-1, 1))
    # labels_mia = [0] * len(retrained_test_losses) + [1] * len(ft_forget_losses)
    # unlearned_mia_scores = simple_mia(retrained_samples_mia, labels_mia)
    # print(f"MIA unlearned vs retrained {unlearned_mia_scores.mean():.3f}")


    # print("F1:", accuracy(unlearned_model, retain_loader)/ accuracy(rt_model, retain_loader))
    # print("F2:", accuracy(unlearned_model, test_loader)/ accuracy(rt_model, test_loader))


if __name__ == "__main__":
    main()
