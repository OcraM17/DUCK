import torch 
import os
from error_propagation import Complex

class OPT:
    run_name = "test"
    dataset = 'cifar10'
    seed = [0]#[0,1,2,3,4,5,6,7,8,42]
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    class_to_be_removed = None#[i*10 for i in range(10)] #[0,1] ##,6,7,8
    save_model = True
    save_df = True
    load_unlearned_model = False
    if class_to_be_removed is None:
        mode = "HR"
    else:
        mode = "CR"

    # gets current folder path
    root_folder = os.path.dirname(os.path.abspath(__file__)) + "/"

    # Model
    model = 'resnet18'#'AllCNN'

    ### RUN model type
    run_original = False
    run_unlearn = True
    run_rt_model = False
    
    # Data
    data_path = '~/data'
    if dataset == 'cifar10':
        num_classes = 10
        batch_fgt_ret_ratio = 1
    elif dataset == 'cifar100':
        num_classes = 100
        batch_fgt_ret_ratio = 5
    elif dataset == 'tinyImagenet':
        num_classes = 200
        batch_fgt_ret_ratio = 30
    elif dataset == 'VGG':
        num_classes = 10
    
    
    num_workers = 8

    name_competitor = "NegativeGradient"#'CBCR' #NegativeGradient, RandomLabels,         # Amnesiac, Hiding...
    
    # unlearning params
    #set class to be remove to None if you want to unlearn a set of samples that belong to different classes
    batch_size = 256
    epochs_unlearn = 30 #best 5
    lr_unlearn = 0.06#cifar100 #0.0001#0.0000005 #best 0.001
    wd_unlearn = 0
    momentum_unlearn = 0.9
    temperature = 2

    #CBCR specific
    lambda_1 = 1.#1#.5#cifar100 .1#vgg subj
    lambda_2 = 1.4#1.5 #0.5#cifar100 1#vgg subj
    target_accuracy = 0.01 #0.76 cifar100 
    

    ###MLP
    iter_MIA = 1 #numo f iterations
    verboseMIA = False

   
    if model== 'resnet18':
        if dataset== 'cifar100':
            or_model_weights_path = root_folder+'weights/Final_CIFAR100_Resnet18.pth'
            weight_file_id = '1pksj54mSsaDdkwSh1V9KA_SP7ZkafkVI'
            weight_file_id_RT = '1QizS5_YTNmsfgvVw0a2H9HSU8xuzBG8N'

            if mode == "HR":
                RT_model_weights_path = root_folder+'weights/chks_cifar100/best_checkpoint_without_5000.pth'
            else:
                RT_model_weights_path = root_folder+f'weights/chks_cifar100/best_checkpoint_without_{class_to_be_removed}.pth'
        
        elif dataset== 'cifar10':
            or_model_weights_path = root_folder+'weights/Final_CIFAR10_Resnet18.pth'
            weight_file_id = '198mmeueWTdH66eTlE0vJu0lLd72nQBbr'
            weight_file_id_RT = '1URa2nH_IyAUzIUgv_ICEdm-5b1w-Zrf0'

            if mode == "HR":
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
    

    a_or = {
        "cifar10" : [Complex(88.72, 0.28)/100.,Complex(88.64, 0.63)/100.], #[0] HR, [1] CR 
        "cifar100" : [Complex(77.56, 0.29)/100., Complex(77.55, 0.11)/100.],
        "tinyImagenet" : [Complex(68.22, 0.54)/100.,Complex(68.40, 0.07)/100.]

    }
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
