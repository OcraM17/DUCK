import os
import numpy as np
import requests
import torch
from torchvision.models.resnet import resnet18,ResNet18_Weights
#from resnet import resnet18 
from sklearn import linear_model, model_selection
import torch.nn as nn
from opts import OPT as opt
import torchvision
from allcnn import AllCNN
import pickle as pk
import tqdm
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support
import os
import zipfile
from competitors import *
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

def choose_competitor(name):
    if name=='FineTuning':
        return FineTuning
    elif name=='NegativeGradient':
        return NegativeGradient
    elif name=='RandomLabels':
        return RandomLabels
    elif name=='Amnesiac':
        return Amnesiac
    elif name=='Hidden':
        return Hiding
    

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


def get_resnet18_trained():

    local_path = opt.or_model_weights_path
    if not os.path.exists(local_path):
        download_weights_drive(local_path,opt.weight_file_id,opt.root_folder)

    weights_pretrained = torch.load(local_path)
    model = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    if opt.dataset != 'cifar10':
        model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        model.maxpool = nn.Identity()
        model.fc = nn.Sequential(nn.Dropout(0), nn.Linear(512, opt.num_classes)) 
    else:
        model.fc = nn.Linear(512, opt.num_classes)

    model.load_state_dict(weights_pretrained)

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

########################
#OLD STUFF
# ########################


# def get_allcnn_trained_on_cifar10():
#     weight_path = "./checkpoints/main_epoch0078_seed42_acc0.921_BEST.pt"
#     model = AllCNN(num_classes=opt.num_classes)
#     model.load_state_dict(torch.load(weight_path, map_location=opt.device))

#     return model
# def get_retrained_model(opt):
#     # download weights of a model trained exclusively on the retain set
#     if opt.class_to_be_removed is None:
#         local_path = opt.RT_model_weights_path
#         if not os.path.exists(local_path):
#             response = requests.get(
#                 "https://unlearning-challenge.s3.eu-west-1.amazonaws.com/cifar10/" + local_path
#             )
#             open(local_path, "wb").write(response.content)
            
#     else:
#         local_path = '/home/jb/Documents/MachineUnlearning/oracle_cifar10_class_rem.pth'

#     weights_pretrained = torch.load(local_path)

#     # load model with pre-trained weights
#     rt_model = resnet18(weights=None, num_classes=opt.num_classes)
#     rt_model.load_state_dict(weights_pretrained)

#     return rt_model

# def get_resnet18_trained_on_tinyimagenet():

#     local_path = opt.or_model_weights_path
#     if not os.path.exists(local_path):
#         download_weights_drive(local_path,opt.weight_file_id)

#     weights_pretrained = torch.load(local_path)

#     model = torchvision.models.resnet18(pretrained=True)
#     model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#     model.maxpool = nn.Identity()
#     model.fc = nn.Sequential(nn.Dropout(0), nn.Linear(512, 100)) 

#     model.load_state_dict(weights_pretrained)

    # return model

# def get_resnet18_trained_on_cifar10():
    
#     local_path = opt.or_model_weights_path
#     if not os.path.exists(local_path):
#         response = requests.get(
#             "https://unlearning-challenge.s3.eu-west-1.amazonaws.com/weights_resnet18_cifar10.pth"
#         )
#         open(local_path, "wb").write(response.content)

#     weights_pretrained = torch.load(local_path, map_location=opt.device)
#     model = torchvision.models.resnet18(weights=None, num_classes=opt.num_classes)

#     model.load_state_dict(weights_pretrained)

#     return model

# from utils import get_membership_attack_prob, get_MIA_MLP
# def get_outputs(retain,forget,net,filename,opt=opt):
#     bbone = torch.nn.Sequential(*(list(net.children())[:-1] + [torch.nn.Flatten()]))
#     fc=net.fc

#     bbone.eval(), fc.eval()

#     out_all_fgt = None
#     lab_ret_list = []
#     lab_fgt_list = []

#     for (img_ret, lab_ret), (img_fgt, lab_fgt) in zip(retain, forget):
#         img_ret, lab_ret, img_fgt, lab_fgt = img_ret.to(opt.device), lab_ret.to(opt.device), img_fgt.to(opt.device), lab_fgt.to(opt.device)
        
#         logits_fgt = bbone(img_fgt)
#         outputs_fgt = fc(logits_fgt)
        
#         logits_ret = bbone(img_ret)
#         outputs_ret = fc(logits_ret)
        
#         lab_fgt_list.append(lab_fgt)
#         lab_ret_list.append(lab_ret)

#         if out_all_fgt is None:
#             out_all_fgt = outputs_fgt
#             out_all_ret = outputs_ret
#             logits_all_fgt = logits_fgt
#             logits_all_ret = logits_ret


#         else:
#             out_all_fgt = torch.concatenate((out_all_fgt,outputs_fgt),dim=0)
#             out_all_ret = torch.concatenate((out_all_ret,outputs_ret),dim=0)

#             logits_all_fgt = torch.concatenate((logits_all_fgt,logits_fgt),dim=0)
#             logits_all_ret = torch.concatenate((logits_all_ret,logits_ret),dim=0)


#     print('check ACCURACY retain ',torch.sum((torch.argmax(out_all_ret,dim=1))==torch.cat(lab_ret_list))/out_all_ret.shape[0])
#     file = open(filename,'wb')

#     pk.dump([out_all_fgt.detach().cpu(),out_all_ret.detach().cpu(),logits_all_fgt.detach().cpu(),logits_all_ret.detach().cpu(),torch.cat(lab_fgt_list).detach().cpu(),torch.cat(lab_ret_list).detach().cpu()],file)

# class DeepMLP(nn.Module):
#     def __init__(self, num_classes=2, num_layers=3, num_hidden=100):
#         super(DeepMLP, self).__init__()
#         self.num_layers = num_layers
#         self.num_hidden = num_hidden

#         self.fc1 = nn.Linear(100, num_hidden)
#         self.bn1 = nn.BatchNorm1d(num_hidden)
#         self.fully_connected = nn.ModuleList([nn.Linear(num_hidden, num_hidden) for _ in range(num_layers)])
#         self.fc = nn.Linear(num_hidden, num_classes)

#         self.bnorms = nn.ModuleList([nn.BatchNorm1d(num_hidden) for _ in range(num_layers)])


#     def forward(self, x):
#         x = nn.functional.relu(self.bn1(self.fc1(x)))
#         for i in range(self.num_layers):
#             x = self.fully_connected[i](x)
#             x = self.bnorms[i](x)
#             x = nn.functional.relu(x)
#         x = self.fc(x)

#         return x
    
# def plot_MIA(dict_acc,name ='MIA_test'):
   
#     mean = np.asarray([dict_acc['test_mean'],dict_acc['test_retain'],dict_acc['test_fgt']])
#     std = np.asarray([dict_acc['test_std'],dict_acc['test_retain_std'],dict_acc['test_fgt_std']])
    
#     plt.errorbar(np.arange(3),mean,std)
#     plt.ylabel('Accuracy')
#     #set x labels to test retain and forget
#     plt.xticks(np.arange(3),['test','retain','forget'])
#     plt.xlabel('case')
#     plt.title(f'{name}')
#     plt.savefig(f'{name}.png')
#     plt.close()

# def get_MIA(dataloader,model,MLP,weights,opt,dict_acc,verbose=False,case='fgt'):
#     model.eval()
#     MLP.eval()
#     MLP = MLP.to(opt.device)
#     correct = torch.zeros((100,))
#     total = torch.zeros((100,))
    
#     with torch.no_grad():
#         for jj,data in enumerate(dataloader):
#             images_all, labels_all = data

#             for i in range(100):
#                 MLP.load_state_dict(weights[f'class_{i}'])
#                 images_all, labels_all = images_all.to(opt.device), labels_all.to(opt.device)
#                 images, labels = images_all[labels_all==i], labels_all[labels_all==i]
#                 if images.shape[0] == 0:
#                     continue
#                 else:
#                     outputs = model(images)

#                     predictions = MLP(torch.nn.functional.softmax(outputs,dim=1))
#                     _, predicted = torch.max(predictions, 1)
#                     total[i] += labels.size(0)
#                     correct[i] += (predicted.detach().cpu() == 0).sum().item()


#     for i in range(100):
#         if verbose: print(f'Class {i} get MIA precision {round(correct / total,3)}')
#         dict_acc[f'{case}_class_{i}'] = correct[i] / total[i]


#     accuracies = []
#     for key, value in dict_acc.items():
#         accuracies.append(value)

#     accuracies = np.array(accuracies)
#     print(f'Mean MIA precision for case {case}: {round(np.mean(accuracies),3)}')
#     dict_acc[f'{case}_mean'] = np.mean(accuracies)
#     dict_acc[f'{case}_std'] = np.std(accuracies)
    

