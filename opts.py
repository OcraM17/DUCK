import torch 
import os
class OPT:
    run_name = "test"
    dataset = 'cifar100'
    seed = 42
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # gets current folder path
    root_folder = os.path.dirname(os.path.abspath(__file__)) + "/"

    # Model
    model = 'resnet18'#'AllCNN'
    ### RUN model type
    run_original = True
    run_unlearn = True
    run_rt_model = False
    
    # Data
    data_path = '~/data'
    if dataset == 'cifar10':
        num_classes = 10
    elif dataset == 'cifar100':
        num_classes = 100
    elif dataset == 'tinyImagenet':
        num_classes = 200
    
    num_workers = 4

    
    # unlearning params
    #set class to be remove to None if you want to unlearn a set of samples that belong to different classes
    batch_size = 128
    class_to_be_removed = 5,6,7,8
    epochs_unlearn = 2000 #best 5
    lr_unlearn = 0.001#cifar100
    #0.0001#0.0000005 #best 0.001
    wd_unlearn = 0.
    momentum_unlearn = 0.
    lambda_1 = 1. #1#cifar100 .1#vgg subj
    lambda_2 = 0.7 #0.5#cifar100 1#vgg subj
    target_accuracy = 0.67 #0.76 cifar100


    ###MLP
    iter_MLP = 1 #numo f iterations
    num_layers_MLP=3
    num_epochs_MLP=90
    lr_MLP=0.001
    weight_decay_MLP = 0
    batch_size_MLP=128
    num_hidden_MLP=100
    verboseMLP = False

    if model== 'resnet18':
        if dataset== 'cifar100':
            model_weights = root_folder+'weights/Final_CIFAR100_Resnet18.pth'
            link_weights = '1pksj54mSsaDdkwSh1V9KA_SP7ZkafkVI'
        
        elif dataset== 'cifar10':
            model_weights = root_folder+'weights/Final_CIFAR10_Resnet18.pth'
            link_weights = '198mmeueWTdH66eTlE0vJu0lLd72nQBbr'
            if class_to_be_removed is None:
                model_weights_RT = root_folder+'weights/chks_cifar10/best_checkpoint_without_5000.pth'
                #fare download del file zip e unzipparlo 
                # https://drive.google.com/file/d/1URa2nH_IyAUzIUgv_ICEdm-5b1w-Zrf0/view?usp=drive_link
                link_weights_RT = '1URa2nH_IyAUzIUgv_ICEdm-5b1w-Zrf0'
            else:
                model_weights_RT = root_folder+f'weights/chks_cifar10/best_checkpoint_without_{class_to_be_removed}.pth'
                link_weights_RT = '1IhuzENiHGDuFtb5cDShAhgXdfujlq-lK'
        elif dataset== 'tinyImagenet':
            model_weights = root_folder+'weights/best_model_tiny.pth'
            link_weights = '11wMtPzADDxBsRKBctK0BSJa3jhg48jK4'
            
        elif dataset== 'VGG':
            #to fix
            model_weights = '/home/jb/Documents/MachineUnlearning/weights/resnet18-184-bestXXX.pth'
    else:
        raise NotImplementedError

    #class rem
    #                    cifar10   cifar100
    # lr_unlearn =       0.0001    0.0001       #0.0000005 #best 0.001
    #wd_unlearn =        0.
    #momentum_unlearn =  0.
    #lambda_1 =          1         1            #cifar100 .1#vgg subj
    #lambda_2 =          0.5       0.5          #cifar100 1#vgg subj
    #target_accuracy =   0.01      0.01  


    # random elements rem
    #                    cifar10   cifar100     TinyImagenet
    # lr_unlearn =       0.0001    0.0001       0.0001  #best 0.001
    #wd_unlearn =        0.        0
    #momentum_unlearn =  0.        0
    #lambda_1 =          1         1            1       .1#vgg subj
    #lambda_2 =          0.5       0.7          0.7      1#vgg subj
    #target_accuracy =   0.87      0.76         0.67
