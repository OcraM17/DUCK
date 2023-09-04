
import os
import requests
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, model_selection

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

import torchvision
from torchvision import transforms
from torchvision.utils import make_grid
from torchvision.models import resnet18

from tqdm import tqdm
from copy import deepcopy
from utils import get_dsets, accuracy, compute_losses, simple_mia, get_retrained_model
from unlearn import unlearning, split_network, merge_network

def main():
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    RNG = torch.Generator().manual_seed(SEED)
    np.random.seed(SEED)

    ##### GET DATA #####
    train_loader, test_loader, forget_loader, retain_loader = get_dsets(RNG)


    ##### GET MODEL #####
    local_path = "weights_resnet18_cifar10.pth"
    if not os.path.exists(local_path):
        response = requests.get(
            "https://unlearning-challenge.s3.eu-west-1.amazonaws.com/weights_resnet18_cifar10.pth"
        )
        open(local_path, "wb").write(response.content)

    weights_pretrained = torch.load(local_path, map_location=DEVICE)

    # load model with pre-trained weights
    model = resnet18(weights=None, num_classes=10)
    model.load_state_dict(weights_pretrained)
    model.to(DEVICE)
    model.eval()


    print(f"Train set accuracy: {100.0 * accuracy(DEVICE, model, train_loader):0.1f}%")
    print(f"Test set accuracy: {100.0 * accuracy(DEVICE, model, test_loader):0.1f}%")


    ##### UNLEARN #####
    ft_model = resnet18(weights=None, num_classes=10)
    ft_model.load_state_dict(weights_pretrained)
    ft_model.to(DEVICE)
    ft_model_original = deepcopy(ft_model)

    best_acc = -1
    best_model_idx = -1
    for layer_idx in [0]:#,4,5,6,7]:

        print(f'------ {layer_idx} ------')

        # Execute the unlearing routine. This might take a few minutes.
        # If run on colab, be sure to be running it on  an instance with GPUs
        new_ft_model = unlearning(DEVICE, ft_model, layer_idx, retain_loader, forget_loader, test_loader)
        
        # evaluate new model and save the best one
        current_accuracy = accuracy(DEVICE, new_ft_model, retain_loader)
        if current_accuracy > best_acc:
            best_acc = current_accuracy
            best_model_idx = layer_idx 
            best_model = deepcopy(new_ft_model)

        print(f"Retain set accuracy: {100.0 * accuracy(DEVICE, new_ft_model, retain_loader):0.1f}%")
        print(f"Test set accuracy: {100.0 * accuracy(DEVICE, new_ft_model, test_loader):0.1f}%")



    # BEST UNLEARNED LAYER
    print(f"Best model {best_model_idx} accuracy: {100.0 * accuracy(DEVICE, best_model, test_loader):0.1f}%")
    ft_model = best_model


    # ORIGINAL MODEL
    ft_forget_losses = compute_losses(DEVICE, ft_model, forget_loader)
    ft_test_losses = compute_losses(DEVICE, ft_model, test_loader)
    ft_samples_mia = np.concatenate((ft_test_losses, ft_forget_losses)).reshape((-1, 1))
    labels_mia = [0] * len(ft_test_losses) + [1] * len(ft_forget_losses)
    ft_mia_scores = simple_mia(ft_samples_mia, labels_mia)
    print(f"The MIA has an accuracy of {ft_mia_scores.mean():.3f} on forgotten vs unseen images")


    # RETRAINED MODEL ON RETAIN SET
    rt_model = get_retrained_model(DEVICE, retain_loader, forget_loader)
    rt_test_losses = compute_losses(DEVICE, rt_model, test_loader)
    rt_forget_losses = compute_losses(DEVICE, rt_model, forget_loader)
    rt_samples_mia = np.concatenate((rt_test_losses, rt_forget_losses)).reshape((-1, 1))
    labels_mia = [0] * len(rt_test_losses) + [1] * len(rt_forget_losses)
    rt_mia_scores = simple_mia(rt_samples_mia, labels_mia)
    print(f"RETRAINED MODEL: The MIA has an accuracy of {rt_mia_scores.mean():.3f} on forgotten vs unseen images")


    print(f"Re-trained model.\nAttack accuracy: {rt_mia_scores.mean():0.2f}")
    print(f"Unlearned.\nAttack accuracy: {ft_mia_scores.mean():0.2f}")




if __name__ == "__main__":
    main()