import torch 
import os
class OPT:

    dataset = 'cifar100'
    seed = 42
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    #root_folder = '/home/luigi/Work/MachineUnlearning/'
    # gets current folder path
    root_folder = os.path.dirname(os.path.abspath(__file__)) + "/"



    # Model
    model = 'resnet18'#'AllCNN'
    if model== 'resnet18':
        if dataset== 'cifar100':
            model_weights = root_folder+'weights/Final_CIFAR100_Resnet18.pth'
            link_weights = '1pksj54mSsaDdkwSh1V9KA_SP7ZkafkVI'
        elif dataset== 'cifar10':
            model_weights = root_folder+'weights/Final_CIFAR10_Resnet18.pth'

        elif dataset== 'TinyImagenet':
            model_weights = root_folder+'weights/best_model_tiny.pth'
            link_weights = '11wMtPzADDxBsRKBctK0BSJa3jhg48jK4'
            
        elif dataset== 'VGG':
            #to fix
            model_weights = '/home/jb/Documents/MachineUnlearning/weights/resnet18-184-bestXXX.pth'
    else:
        raise NotImplementedError
    
    # Data
    data_path = '~/data'
    num_classes = 100
    num_workers = 4
    batch_size = 128
    
    # unlearning params
    #set class to be remove to None if you want to unlearn a set of samples that belong to different classes
    class_to_be_removed = 5#None#5 
    epochs_unlearn = 2000 #best 5
    lr_unlearn = 0.00005#cifar100
    #0.0001#0.0000005 #best 0.001
    wd_unlearn = 0.
    momentum_unlearn = 0.
    lambda_1 = 1 #1#cifar100 .1#vgg subj
    lambda_2 = 0.5 #0.5#cifar100 1#vgg subj
    target_accuracy = 0.02 #0.76 cifar100

    # finetuning params
    epochs_fine_tune = 5
    batch_size_FT = 16
    lr_fine_tune = 0.01
    wd_fine_tune = 5e-4
    momentum_fine_tune = 0.9

    ###MLP
    iter_MLP = 5 #numo f iterations
    num_layers_MLP=3
    num_epochs_MLP=90
    lr_MLP=0.001
    weight_decay_MLP = 0
    batch_size_MLP=128
    num_hidden_MLP=100
    verbose_MLP = False