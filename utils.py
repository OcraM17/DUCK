import os
import numpy as np
import requests
import torch
from torchvision.models.resnet import resnet18,ResNet18_Weights,resnet34,ResNet34_Weights,resnet50,ResNet50_Weights
from models import ViT
#from resnet import resnet18 
from sklearn import linear_model, model_selection
import torch.nn as nn
from opts import OPT as opt
import torchvision
from models.allcnn import AllCNN
import pickle as pk
import tqdm
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support
import os
import zipfile
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive


    

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



# Initialize GoogleDrive instance with the credentials
def download_weights_drive(model_weights_path, weight_file_id, root_folder):
    # Specify the path to your JSON key file
    json_key_file_path = root_folder+'client_secrets.json'

    # Check if the JSON key file exists
    if not os.path.isfile(json_key_file_path):
        print(f"Error: JSON key file not found at '{json_key_file_path}'")
        exit(1)
    # Set the GOOGLE_DRIVE_SETTINGS environment variable to the JSON key file path
    os.environ['GOOGLE_DRIVE_SETTINGS'] = json_key_file_path
    # Initialize GoogleAuth instance
    gauth = GoogleAuth()

    # Perform user authentication using LocalWebserverAuth
    gauth.LocalWebserverAuth()

    # Create GoogleDrive instance
    drive = GoogleDrive(gauth)

    # Set the ID of the file in your Google Drive
    file_id = weight_file_id  # Replace with the actual file ID

    # Set the path to save the downloaded file
    download_path = model_weights_path  # Replace with your desired local file path

    # Download the file
    file = drive.CreateFile({'id': file_id})
    file.GetContentFile(download_path)

    print(f"File '{file['title']}' downloaded to '{download_path}'")


def get_retrained_model():

    local_path = opt.RT_model_weights_path
    #DOWNLOAD ZIP
    if not os.path.exists(local_path):
        download_weights_drive("rt_models.zip", opt.weight_file_id_RT, opt.root_folder)
        unzip_file(opt.root_folder + "rt_models.zip", opt.root_folder + "weights/.")
        os.system("rm rt_models.zip")
    weights_pretrained = torch.load(local_path)
    model = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    if opt.dataset != 'cifar10':
        model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        model.maxpool = nn.Identity()
        model.fc = nn.Sequential(nn.Dropout(0), nn.Linear(512, opt.num_classes)) 
    else:
        model.fc = nn.Sequential(nn.Dropout(0), nn.Linear(512, opt.num_classes))

    model.load_state_dict(weights_pretrained)

    return model


def get_resnet_trained():

    local_path = opt.or_model_weights_path
    print('LOAD weights: ', local_path)
    if not os.path.exists(local_path):
        download_weights_drive(local_path,opt.weight_file_id,opt.root_folder)

    weights_pretrained = torch.load(local_path)
    if opt.model=='resnet18':
        model = torchvision.models.resnet18(weights=None)
    elif opt.model=='resnet34':
        model = torchvision.models.resnet34(weights=None)
    elif opt.model=='resnet50':
        model = torchvision.models.resnet50(weights=None)

    if opt.dataset != 'cifar10':
        model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        model.maxpool = nn.Identity()
        model.fc = nn.Sequential(nn.Dropout(0), nn.Linear(model.fc.in_features, opt.num_classes)) 
    else:
        model.fc = nn.Linear(512, opt.num_classes)
    
    #print(weights_pretrained)
    model.load_state_dict(weights_pretrained)

    return model

def get_ViT_trained():
    local_path = opt.or_model_weights_path
    if not os.path.exists(local_path):
        download_weights_drive(local_path,opt.weight_file_id,opt.root_folder)
    
    weights_pretrained = torch.load(local_path)
    if opt.dataset == 'cifar10' or opt.dataset == 'cifar100':
        image_size=32
    elif opt.dataset == 'tinyImagenet':
        image_size = 64

    model = ViT.ViT(image_size=image_size, patch_size=4, num_classes=opt.num_classes, dim=512, depth=8, heads=12, mlp_dim=512, pool = 'cls', channels = 3, dim_head = 128, dropout = 0, emb_dropout = 0)

    model.load_state_dict(weights_pretrained)

    return model

def get_AllCNN_trained():
    local_path = opt.or_model_weights_path
    if not os.path.exists(local_path):
        download_weights_drive(local_path,opt.weight_file_id,opt.root_folder)
    
    weights_pretrained = torch.load(local_path)
    model = AllCNN()


def get_trained_model():
    if 'resnet' in opt.model:
        model = get_resnet_trained()
    elif 'ViT' in opt.model:
        model = get_ViT_trained()
    elif opt.model == 'AllCNN':
        model = get_AllCNN_trained()
    return model
##############################################################################################################




def get_resnet50_trained_on_VGGFace():
    local_path = "/home/jb/Documents/MachineUnlearning/weights/net_weights_resnet50_VGG.pth"
    weights_pretrained = torch.load(local_path, map_location=opt.device)

    # load model with pre-trained weights
    model =torchvision.models.resnet50(weights=None)
    # Change the final layer
    model.fc = nn.Sequential(nn.Dropout(p=0.0),nn.Linear(model.fc.in_features, 9))

    model.load_state_dict(weights_pretrained)
    return model

def get_resnet50_trained_on_VGGFace_10_subjects():
    #merge with the function above
    local_path = "/home/jb/Documents/MachineUnlearning/weights/net_weights_resnet50_VGG_10sub.pth"
    weights_pretrained = torch.load(local_path, map_location=opt.device)

    # load model with pre-trained weights
    model =torchvision.models.resnet50(weights=None)
    # Change the final layer
    model.fc = nn.Sequential(nn.Dropout(p=0.0),nn.Linear(model.fc.in_features, 10))

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


def get_outputs(retain,forget,net,filename,opt=opt):
    bbone = torch.nn.Sequential(*(list(net.children())[:-1] + [torch.nn.Flatten()]))
    fc=net.fc

    bbone.eval(), fc.eval()

    out_all_fgt = None
    out_all_ret = None
    lab_ret_list = []
    lab_fgt_list = []

    for img_ret, lab_ret in retain:
        
        img_ret, lab_ret = img_ret.to(opt.device), lab_ret.to(opt.device)
        
        logits_ret = bbone(img_ret)
        outputs_ret = fc(logits_ret)
        
        lab_ret_list.append(lab_ret)

        if out_all_ret is None:

            out_all_ret = outputs_ret
            logits_all_ret = logits_ret

        else:
            out_all_ret = torch.concatenate((out_all_ret,outputs_ret),dim=0)
            logits_all_ret = torch.concatenate((logits_all_ret,logits_ret),dim=0)


    for img_fgt, lab_fgt in forget:

        img_fgt, lab_fgt = img_fgt.to(opt.device), lab_fgt.to(opt.device)
    
        logits_fgt = bbone(img_fgt)
        outputs_fgt = fc(logits_fgt)
        

        
        lab_fgt_list.append(lab_fgt)

        if out_all_fgt is None:
            out_all_fgt = outputs_fgt

            logits_all_fgt = logits_fgt



        else:
            out_all_fgt = torch.concatenate((out_all_fgt,outputs_fgt),dim=0)
            logits_all_fgt = torch.concatenate((logits_all_fgt,logits_fgt),dim=0)



    print('check ACCURACY retain ',torch.sum((torch.argmax(out_all_ret,dim=1))==torch.cat(lab_ret_list))/out_all_ret.shape[0])
    file = open(filename,'wb')

    pk.dump([out_all_fgt.detach().cpu(),out_all_ret.detach().cpu(),logits_all_fgt.detach().cpu(),logits_all_ret.detach().cpu(),torch.cat(lab_fgt_list).detach().cpu(),torch.cat(lab_ret_list).detach().cpu()],file)

def unzip_file(file_path, destination_path):
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(destination_path)
