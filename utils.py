import os
import numpy as np
import requests
import torch
from torchvision.models.resnet import resnet18
from sklearn import linear_model, model_selection
import torch.nn as nn
from opts import OPT as opt

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    # deterministic cudnn
    torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False


def accuracy(net, loader):
    """Return accuracy on a dataset given by the data loader."""
    correct = 0
    total = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(opt.device), targets.to(opt.device)
        outputs = net(inputs)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    return correct / total


def compute_losses(net, loader):
    """Auxiliary function to compute per-sample losses"""

    criterion = nn.CrossEntropyLoss(reduction="none")
    all_losses = []

    for inputs, targets in loader:
        inputs, targets = inputs.to(opt.device), targets.to(opt.device)

        logits = net(inputs)
        losses = criterion(logits, targets).numpy(force=True)
        for l in losses:
            all_losses.append(l)

    return np.array(all_losses)



def simple_mia(sample_loss, members, n_splits=10, random_state=0):
    """Computes cross-validation score of a membership inference attack.

    Args:
      sample_loss : array_like of shape (n,).
        objective function evaluated on n samples.
      members : array_like of shape (n,),
        whether a sample was used for training.
      n_splits: int
        number of splits to use in the cross-validation.
    Returns:
      scores : array_like of size (n_splits,)
    """

    unique_members = np.unique(members)
    if not np.all(unique_members == np.array([0, 1])):
        raise ValueError("members should only have 0 and 1s")

    attack_model = linear_model.LogisticRegression()
    cv = model_selection.StratifiedShuffleSplit(
        n_splits=n_splits, random_state=random_state
    )
    return model_selection.cross_val_score(
        attack_model, sample_loss, members, cv=cv, scoring="accuracy"
    )



def get_retrained_model(retain_loader, forget_loader):
    # download weights of a model trained exclusively on the retain set
    local_path = "retrain_weights_resnet18_cifar10.pth"
    if not os.path.exists(local_path):
        response = requests.get(
            "https://unlearning-challenge.s3.eu-west-1.amazonaws.com/cifar10/" + local_path
        )
        open(local_path, "wb").write(response.content)

    weights_pretrained = torch.load(local_path, map_location=opt.device)

    # load model with pre-trained weights
    rt_model = resnet18(weights=None, num_classes=10)
    rt_model.load_state_dict(weights_pretrained)
    rt_model.to(opt.device)
    rt_model.eval()

    # print its accuracy on retain and forget set
    print(f"[ Train ] ret: {accuracy(rt_model, retain_loader):.3f}  fgt: {accuracy(rt_model, forget_loader):.3f}")

    return rt_model

def get_resnet18_trained_on_cifar10():
    local_path = "weights_resnet18_cifar10.pth"
    if not os.path.exists(local_path):
        response = requests.get(
            "https://unlearning-challenge.s3.eu-west-1.amazonaws.com/weights_resnet18_cifar10.pth"
        )
        open(local_path, "wb").write(response.content)

    weights_pretrained = torch.load(local_path, map_location=opt.device)

    # load model with pre-trained weights
    model = resnet18(weights=None, num_classes=10)

    model.load_state_dict(weights_pretrained)

    return model



def compute_metrics(model, train_loader, forget_loader, retain_loader, all_val_loader, val_fgt_loader, val_retain_loader):

    # compute losses for the original forget set
    main_forget_losses = compute_losses(model, forget_loader)

    losses = compute_losses(model, all_val_loader)
    samples_mia = np.concatenate((losses, main_forget_losses)).reshape((-1, 1))
    labels_mia = [0] * len(losses) + [1] * len(main_forget_losses)
    mia_scores = simple_mia(samples_mia, labels_mia)

    # fgt
    fgt_losses = compute_losses(model, val_fgt_loader)
    fgt_samples_mia = np.concatenate((fgt_losses, main_forget_losses)).reshape((-1, 1))
    labels_mia = [0] * len(fgt_losses) + [1] * len(main_forget_losses)
    fgt_mia_scores = simple_mia(fgt_samples_mia, labels_mia)

    # retain
    retain_losses = compute_losses(model, val_retain_loader)
    retain_samples_mia = np.concatenate((retain_losses, main_forget_losses)).reshape((-1, 1))
    labels_mia = [0] * len(retain_losses) + [1] * len(main_forget_losses)
    retain_mia_scores = simple_mia(retain_samples_mia, labels_mia)
    
    
    #print(f"[ MIA - val ] all:{mia_scores.mean():.3f}  fgt:{fgt_mia_scores.mean():.3f}  ret:{retain_mia_scores.mean():.3f}")


    print(f"[ ACC-train ] all:{accuracy(model, train_loader):.3f}  fgt: {accuracy(model, forget_loader):.3f} ret: {accuracy(model, retain_loader):.3f}  ")
    print(f"[ ACC - val ] all:{accuracy(model, all_val_loader):.3f}  fgt:{accuracy(model, val_fgt_loader):.3f}  ret:{accuracy(model, val_retain_loader):.3f}")
