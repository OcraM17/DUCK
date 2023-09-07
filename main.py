
import os
import requests
import numpy as np
import torch
from torchvision.models import resnet18
from copy import deepcopy
from dsets import get_dsets_remove_class
from utils import accuracy, compute_metrics, get_resnet18_trained_on_cifar10, set_seed
from unlearn import unlearning, fine_tune
from opts import OPT as opt


def main():
    # set random seed
    set_seed(opt.seed)

    ##### GET DATA #####
    train_loader, forget_loader, retain_loader, val_fgt_loader, val_retain_loader, all_val_loader = get_dsets_remove_class(opt.class_to_be_removed)

    ##### GET MODEL #####
    original_pretr_model = get_resnet18_trained_on_cifar10()
    original_pretr_model.to(opt.device)
    original_pretr_model.eval()
    print('\n----ORIGINAL MODEL----')
    print(f"all_train acc: {accuracy(original_pretr_model, train_loader):.3f}")
    print(f"all_val acc: {accuracy(original_pretr_model, all_val_loader):.3f}")

    ##### UNLEARN #####
    print('\n----FIND BEST LAYER----')
    pretr_model = deepcopy(original_pretr_model)
    pretr_model.to(opt.device)
    pretr_model.eval()

    best_acc = -1
    best_layer_idx = -1
    for layer_idx in [0]:#,4,5,6,7]:

        # Execute the unlearing routine. This might take a few minutes.
        # If run on colab, be sure to be running it on  an instance with GPUs
        tmp_unlearned_model = unlearning(pretr_model, layer_idx, retain_loader, forget_loader, all_val_loader)
        
        # evaluate new model and save the best one
        current_accuracy = accuracy(tmp_unlearned_model, retain_loader)
        if current_accuracy > best_acc:
            best_acc = current_accuracy
            best_layer_idx = layer_idx 
            best_unlearned_model = deepcopy(tmp_unlearned_model)

        print(f"[ Layer {layer_idx}   ] ret: {accuracy(tmp_unlearned_model, retain_loader):.3f}  fgt: {accuracy(tmp_unlearned_model, forget_loader):.3f}")

    # BEST UNLEARNED LAYER
    unlearned_model = best_unlearned_model

    print('\n----- UNLEARNED ----')
    compute_metrics(unlearned_model, train_loader, forget_loader, retain_loader, all_val_loader, val_fgt_loader, val_retain_loader)

    print('\n-----FT on RETAIN----')
    unlearned_model_finetuned = fine_tune(unlearned_model, retain_loader, best_layer_idx)
    compute_metrics(unlearned_model_finetuned, train_loader, forget_loader, retain_loader, all_val_loader, val_fgt_loader, val_retain_loader)



    # print('\n----RETRAINED on RETAIN ----')
    # # RETRAINED MODEL ON RETAIN SET
    # # WE SHOULD RETRAIN FROM SCRATCH 
    # rt_model = get_retrained_model(retain_loader, forget_loader) #<--- WATCHOUT THERE IS A PROBLEM HERE THEY LOAD THE MODEL WITH THE WRONG WEIGHTS
    # compute_metrics(rt_model, forget_loader, all_val_loader, val_fgt_loader, val_retain_loader)


if __name__ == "__main__":
    main()
