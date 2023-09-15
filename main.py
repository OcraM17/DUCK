
import os
import requests
import numpy as np
import sys
import torch
from torchvision.models.resnet import resnet18
from copy import deepcopy
from dsets import get_dsets_remove_class, get_dsets
from utils import accuracy, compute_metrics,get_resnet18_trained_on_cifar10, set_seed, compute_losses, simple_mia, get_retrained_model
from unlearn import unlearning, fine_tune, unlearning2, fine_tune2
from opts import OPT as opt

import pickle as pk

def get_outputs(retain,forget,net,filename,opt=opt):
    bbone = torch.nn.Sequential(*(list(net.children())[:-1] + [torch.nn.Flatten()]))
    fc=net.fc

    bbone.eval(), fc.eval()

    out_all_fgt = None
    lab_ret_list = []
    lab_fgt_list = []

    for (img_ret, lab_ret), (img_fgt, lab_fgt) in zip(retain, forget):
        img_ret, lab_ret, img_fgt, lab_fgt = img_ret.to(opt.device), lab_ret.to(opt.device), img_fgt.to(opt.device), lab_fgt.to(opt.device)
        
        logits_fgt = bbone(img_fgt)
        outputs_fgt = fc(logits_fgt)
        
        logits_ret = bbone(img_ret)
        outputs_ret = fc(logits_ret)
        
        lab_fgt_list.append(lab_fgt)
        lab_ret_list.append(lab_ret)

        if out_all_fgt is None:
            out_all_fgt = outputs_fgt
            out_all_ret = outputs_ret
            logits_all_fgt = logits_fgt
            logits_all_ret = logits_ret


        else:
            out_all_fgt = torch.concatenate((out_all_fgt,outputs_fgt),dim=0)
            out_all_ret = torch.concatenate((out_all_ret,outputs_ret),dim=0)

            logits_all_fgt = torch.concatenate((logits_all_fgt,logits_fgt),dim=0)
            logits_all_ret = torch.concatenate((logits_all_ret,logits_ret),dim=0)


    print('check ACCURACY retain ',torch.sum((torch.argmax(out_all_ret,dim=1))==torch.cat(lab_ret_list))/out_all_ret.shape[0])
    file = open(filename,'wb')

    pk.dump([out_all_fgt.detach().cpu(),out_all_ret.detach().cpu(),logits_all_fgt.detach().cpu(),logits_all_ret.detach().cpu(),torch.cat(lab_fgt_list).detach().cpu(),torch.cat(lab_ret_list).detach().cpu()],file)

def main():
    # set random seed
    set_seed(opt.seed)

    ##### GET DATA #####
    train_loader, test_loader, forget_loader, retain_loader, retain_loader2 = get_dsets()#get_dsets_remove_class(opt.class_to_be_removed)

    ##### GET MODEL #####
    original_pretr_model = get_resnet18_trained_on_cifar10() #get_resnet18_trained_on_cifar10()
    original_pretr_model.to(opt.device)
    original_pretr_model.eval()
    print('\n----ORIGINAL MODEL----')
    print(f"TEST-LOADER:{accuracy(original_pretr_model, test_loader):.3f} \nFORGET-LOADER: {accuracy(original_pretr_model, forget_loader):.3f}\nRETAIN-LOADER: {accuracy(original_pretr_model, retain_loader):.3f}  ")
    ft_forget_losses = compute_losses(original_pretr_model, forget_loader)
    ft_test_losses = compute_losses(original_pretr_model, test_loader)
    ft_samples_mia = np.concatenate((ft_test_losses, ft_forget_losses)).reshape((-1, 1))
    labels_mia = [0] * len(ft_test_losses) + [1] * len(ft_forget_losses)
    ft_mia_scores = simple_mia(ft_samples_mia, labels_mia)
    print(f"The MIA has an accuracy of {ft_mia_scores.mean():.3f} on forgotten vs unseen images")
    #get_outputs(retain_loader,forget_loader,original_pretr_model,'/home/jb/Documents/MachineUnlearning/res_original_model.pkl',opt=opt)


    ##### UNLEARN #####
    pretr_model = deepcopy(original_pretr_model)
    pretr_model.to(opt.device)
    pretr_model.eval()

    
    print('\n----- UNLEARNED ----')
    unlearned_model = unlearning2(pretr_model, retain_loader, forget_loader)
    print(f"TEST-LOADER:{accuracy(unlearned_model, test_loader):.3f} \nFORGET-LOADER: {accuracy(unlearned_model, forget_loader):.3f}\nRETAIN-LOADER: {accuracy(unlearned_model, retain_loader):.3f}  ")
    ft_forget_losses = compute_losses(unlearned_model, forget_loader)
    ft_test_losses = compute_losses(unlearned_model, test_loader)
    ft_samples_mia = np.concatenate((ft_test_losses, ft_forget_losses)).reshape((-1, 1))
    labels_mia = [0] * len(ft_test_losses) + [1] * len(ft_forget_losses)
    ft_mia_scores = simple_mia(ft_samples_mia, labels_mia)
    print(f"The MIA has an accuracy of {ft_mia_scores.mean():.3f} on forgotten vs unseen images")
    #get_outputs(retain_loader,forget_loader,unlearned_model,'/home/jb/Documents/MachineUnlearning/res_unlr_model.pkl',opt=opt)
    
    #compute_metrics(unlearned_model, train_loader, forget_loader, retain_loader, all_val_loader, val_fgt_loader, val_retain_loader)

    #print('\n-----FT on RETAIN----')

    #unlearned_model_finetuned = fine_tune2(unlearned_model, retain_loader2)
    #print(f"TEST-LOADER:{accuracy(unlearned_model_finetuned, test_loader):.3f} \nFORGET-LOADER: {accuracy(unlearned_model_finetuned, forget_loader):.3f}\nRETAIN-LOADER: {accuracy(unlearned_model_finetuned, retain_loader):.3f}  ")
    #ft_forget_losses = compute_losses(unlearned_model_finetuned, forget_loader)
    #ft_test_losses = compute_losses(unlearned_model_finetuned, test_loader)
    #ft_samples_mia = np.concatenate((ft_test_losses, ft_forget_losses)).reshape((-1, 1))
    #labels_mia = [0] * len(ft_test_losses) + [1] * len(ft_forget_losses)
    #ft_mia_scores = simple_mia(ft_samples_mia, labels_mia)
    #print(f"The MIA has an accuracy of {ft_mia_scores.mean():.3f} on forgotten vs unseen images")
    #compute_metrics(unlearned_model_finetuned, train_loader, forget_loader, retain_loader, all_val_loader, val_fgt_loader, val_retain_loader)



    print('\n----RETRAINED on RETAIN ----')
    # # RETRAINED MODEL ON RETAIN SET
    # # WE SHOULD RETRAIN FROM SCRATCH 

    rt_model = get_retrained_model(retain_loader, forget_loader) #<--- WATCHOUT THERE IS A PROBLEM HERE THEY LOAD THE MODEL WITH THE WRONG WEIGHTS
    print(f"TEST-LOADER:{accuracy(rt_model, test_loader):.3f} \nFORGET-LOADER: {accuracy(rt_model, forget_loader):.3f}\nRETAIN-LOADER: {accuracy(rt_model, retain_loader):.3f}  ")
    ft_forget_losses = compute_losses(rt_model, forget_loader)
    ft_test_losses = compute_losses(rt_model, test_loader)

    ft_forget_losses= ft_forget_losses[:len(ft_test_losses)]
    ft_test_losses = ft_test_losses[:len(ft_forget_losses)]

    ft_samples_mia = np.concatenate((ft_test_losses, ft_forget_losses)).reshape((-1, 1))
    labels_mia = [0] * len(ft_test_losses) + [1] * len(ft_forget_losses)
    ft_mia_scores = simple_mia(ft_samples_mia, labels_mia)
    print(f"The MIA has an accuracy of {ft_mia_scores.mean():.3f} on forgotten vs unseen images")


    print("F1:", accuracy(unlearned_model, retain_loader)/ accuracy(rt_model, retain_loader))
    print("F2:", accuracy(unlearned_model, test_loader)/ accuracy(rt_model, test_loader))


if __name__ == "__main__":
    main()
