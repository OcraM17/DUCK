import torch 
import os
import argparse
from error_propagation import Complex

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default="test")
    parser.add_argument("--dataset", type=str, default="cifar100")
    parser.add_argument("--mode", type=str, default="CR")
    parser.add_argument("--cuda", type=int, default=0, help="Select zero-indexed cuda device. -1 to use CPU.")
    
    parser.add_argument("--load_unlearned_model",action='store_true')
    
    parser.add_argument("--save_model", action='store_true')
    parser.add_argument("--save_df", action='store_true')

    parser.add_argument("--run_original", action='store_true')
    parser.add_argument("--run_unlearn", action='store_true')
    parser.add_argument("--run_rt_model", action='store_true')

    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--method", type=str, default="CBCR")

    parser.add_argument("--model", type=str, default='resnet18')
    parser.add_argument("--bsize", type=int, default=256)
    parser.add_argument("--wd", type=float, default=0.0)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--lr", type=float, default=0.0004)
    parser.add_argument("--epochs", type=int, default=200, help='Num of epochs, for unlearning algorithms it is the max num of epochs') # <------- epochs train
    parser.add_argument("--scheduler", type=int, nargs='+', default=[25,40])
    parser.add_argument("--temperature", type=float, default=2)
    parser.add_argument("--lambda_1", type=float, default=1)
    parser.add_argument("--lambda_2", type=float, default=1.4)

    options = parser.parse_args()
    return options


class OPT:
    args = get_args()
    print(args)
    run_name = args.run_name
    dataset = args.dataset

    
    mode = args.mode
    if args.mode == 'HR':
        seed = [0,1,2,3,4,5,6,7,8,42]
        class_to_remove = None
    else:
        seed = [42]
        if dataset == 'cifar10' or dataset=="VGG":
            class_to_remove = [i*1 for i in range(10)]
        elif dataset == 'cifar100':
            class_to_remove = [i*10 for i in range(10)]
        elif dataset == 'tinyImagenet':
            class_to_remove = [i*20 for i in range(10)] 
   
        #class_to_remove = [[i for i in range(100)][:j] for j in [1]+[z*10 for z in range(1,10)]+[98]]
        #print('Class to remove iter. : ', class_to_remove)

    device = f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu"
    
    
    save_model = args.save_model
    save_df = args.save_df
    load_unlearned_model = args.load_unlearned_model
    


    # gets current folder path
    root_folder = os.path.dirname(os.path.abspath(__file__)) + "/"

    # Model
    model = args.model
    ### RUN model type
    run_original = args.run_original
    run_unlearn = args.run_unlearn
    run_rt_model = args.run_rt_model
    
    # Data
    data_path = os.path.expanduser('~/data')
    if dataset == 'cifar10':
        num_classes = 10
        batch_fgt_ret_ratio = 5
    elif dataset == 'cifar100':
        num_classes = 100
        batch_fgt_ret_ratio = 5
    elif dataset == 'tinyImagenet':
        num_classes = 200
        batch_fgt_ret_ratio = 5
    elif dataset == 'VGG':
        num_classes = 10
    
    
    num_workers = args.num_workers

    method = args.method#'CBCR' #NegativeGradient, RandomLabels,         # Amnesiac, Hiding...
    
    # unlearning params
        
    batch_size = args.bsize
    epochs_unlearn = args.epochs
    lr_unlearn = args.lr
    wd_unlearn = args.wd
    momentum_unlearn = args.momentum
    temperature = args.temperature
    scheduler = args.scheduler

    #CBCR specific
    lambda_1 = args.lambda_1
    lambda_2 = args.lambda_2
    target_accuracy = 0.01 
    

    ###MLP
    iter_MIA = 3 #numo f iterations
    verboseMIA = False

   
    if model== 'resnet18':
        if dataset== 'cifar100':
            or_model_weights_path = root_folder+'weights/Final_CIFAR100_Resnet18.pth'
            weight_file_id = '1pksj54mSsaDdkwSh1V9KA_SP7ZkafkVI'
            weight_file_id_RT = '1QizS5_YTNmsfgvVw0a2H9HSU8xuzBG8N'

            if mode == "HR":
                RT_model_weights_path = root_folder+'weights/chks_cifar100/best_checkpoint_without_5000.pth'
            else:
                RT_model_weights_path = root_folder+f'weights/chks_cifar100/best_checkpoint_without_{class_to_remove}.pth'
        
        elif dataset== 'cifar10':
            or_model_weights_path = root_folder+'weights/Final_CIFAR10_Resnet18.pth'
            weight_file_id = '198mmeueWTdH66eTlE0vJu0lLd72nQBbr'
            weight_file_id_RT = '1URa2nH_IyAUzIUgv_ICEdm-5b1w-Zrf0'

            if mode == "HR":
                RT_model_weights_path = root_folder+'weights/chks_cifar10/best_checkpoint_without_5000.pth'
            else:
                RT_model_weights_path = root_folder+f'weights/chks_cifar10/best_checkpoint_without_{class_to_remove}.pth'

        elif dataset== 'tinyImagenet':
            or_model_weights_path = root_folder+'weights/best_model_tiny.pth'
            weight_file_id = '11wMtPzADDxBsRKBctK0BSJa3jhg48jK4'
            
        elif dataset== 'VGG':
            #to fix
            or_model_weights_path = root_folder+'weights/best_model_VGG.pth'
            weight_file_id = '1juieMSI1mEe_SzgDfxL6lzZ-XpD6ZMGF'
            weight_file_id_RT = '101_zAZBLxOp9hn3QkTQwWkvrnO7FPSYy'

            if class_to_remove is None:
                RT_model_weights_path = root_folder+'weights/chks_vgg/chk_VGG_10perc.pth'
            else:
                RT_model_weights_path = root_folder+'weights/chks_vgg/best_model_00.pth'
    
    elif model == 'resnet50':
        if dataset== 'cifar100':
            or_model_weights_path = root_folder+'/chks_cifar100/best_checkpoint_Resnet50.pth'
    
    elif model == 'resnet34':
        if dataset== 'cifar100':
            or_model_weights_path = root_folder+'/chks_cifar100/best_checkpoint_Resnet34.pth'
    
    elif model == 'ViT':
        #raise error not implemented
        raise NotImplementedError
        
    elif model == 'AllCNN':
        if dataset== 'cifar100':
            or_model_weights_path = root_folder+'/chks_cifar100/best_checkpoint_AllCNN.pth'
    else:
        raise NotImplementedError
    

    a_or = {
        "cifar10" : [Complex(88.72, 0.28)/100.,Complex(88.64, 0.63)/100.], #[0] HR, [1] CR 
        "cifar100" : [Complex(77.56, 0.29)/100., Complex(77.55, 0.11)/100.],
        "tinyImagenet" : [Complex(68.22, 0.54)/100.,Complex(68.40, 0.07)/100.],
        "VGG" : [Complex(91.18, 2.92)/100.,Complex(91.18, 2.92)/100.]

    }



    
