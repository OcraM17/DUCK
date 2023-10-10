import torch 
import os
class OPT:
    run_name = "Tuning_MLP"
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
    run_rt_model = True
    
    # Data
    data_path = '~/data'
    if dataset == 'cifar10':
        num_classes = 10
    elif dataset == 'cifar100':
        num_classes = 100
    elif dataset == 'tinyImagenet':
        num_classes = 200
    elif dataset == 'VGG':
        num_classes = 10
    
    
    num_workers = 4

    
    # unlearning params
    #set class to be remove to None if you want to unlearn a set of samples that belong to different classes
    batch_size = 4096
    class_to_be_removed = 0 ##,6,7,8
    epochs_unlearn = 2000 #best 5
    lr_unlearn = 0.001#cifar100 #0.0001#0.0000005 #best 0.001
    wd_unlearn = 0
    momentum_unlearn = 0.9
    lambda_1 = .5 #1#cifar100 .1#vgg subj
    lambda_2 = .5 #0.5#cifar100 1#vgg subj
    target_accuracy = 0.01 #0.76 cifar100 
    unlearning_time = None

    ###MLP
    iter_MLP = 3 #numo f iterations
    num_layers_MLP = 3
    num_epochs_MLP = 120
    lr_MLP = 0.005
    weight_decay_MLP = 0.#00001#001
    batch_size_MLP = 160
    num_hidden_MLP = 80
    verboseMLP = True
    useMLP = False

    #Competitor
    competitor = False
    if competitor:
        name_competitor = 'FineTuning' #NegativeGradient, FineTuning, RandomLabels, Amnesiac, Hiding...
        lr_competitor = 0.1 #FineTuning:0.1, else:0.01
        epochs_competitor = 10
        momentum_competitor = 0.9
        wd_competitor = 5e-4
    else:
        name_competitor, lr_competitor, epochs_competitor, momentum_competitor, wd_competitor = None, None, None, None, None

    if model== 'resnet18':
        if dataset== 'cifar100':
            or_model_weights_path = root_folder+'weights/Final_CIFAR100_Resnet18.pth'
            weight_file_id = '1pksj54mSsaDdkwSh1V9KA_SP7ZkafkVI'
            weight_file_id_RT = '1QizS5_YTNmsfgvVw0a2H9HSU8xuzBG8N'

            if class_to_be_removed is None:
                RT_model_weights_path = root_folder+'weights/chks_cifar100/best_checkpoint_without_5000.pth'
            else:
                RT_model_weights_path = root_folder+f'weights/chks_cifar100/best_checkpoint_without_{class_to_be_removed}.pth'
        
        elif dataset== 'cifar10':
            or_model_weights_path = root_folder+'weights/Final_CIFAR10_Resnet18.pth'
            weight_file_id = '198mmeueWTdH66eTlE0vJu0lLd72nQBbr'
            weight_file_id_RT = '1URa2nH_IyAUzIUgv_ICEdm-5b1w-Zrf0'

            if class_to_be_removed is None:
                RT_model_weights_path = root_folder+'weights/chks_cifar10/best_checkpoint_without_5000.pth'
            else:
                RT_model_weights_path = root_folder+f'weights/chks_cifar10/best_checkpoint_without_{class_to_be_removed}.pth'

        elif dataset== 'tinyImagenet':
            or_model_weights_path = root_folder+'weights/best_model_tiny.pth'
            weight_file_id = '11wMtPzADDxBsRKBctK0BSJa3jhg48jK4'
            
        elif dataset== 'VGG':
            #to fix
            or_model_weights_path = '/home/node002/Documents/MachineUnlearning/chks_vgg/best_model_VGG.pth'
            if class_to_be_removed is None:
                RT_model_weights_path = '/home/node002/Documents/MachineUnlearning/chks_vgg/chk_VGG_10perc.pth'
            else:
                RT_model_weights_path = "/home/node002/Documents/MachineUnlearning/chks_vgg/best_model_00.pth"
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
