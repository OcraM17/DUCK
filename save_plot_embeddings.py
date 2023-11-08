from copy import deepcopy
from dsets import get_dsets_remove_class, get_dsets
#to clean up
from utils import accuracy, set_seed, get_retrained_model,get_resnet50_trained_on_VGGFace_10_subjects,get_resnet18_trained

from unlearn import unlearning
from MIA_code.MIA import get_MIA_MLP
from opts import OPT as opt
import torch.nn as nn
#from publisher import push_results
import time
from utils import choose_competitor,get_outputs

def main():
    # set random seed
    set_seed(opt.seed)
    

    ##### GET DATA #####
    if opt.class_to_be_removed is None:
        train_loader, test_loader, train_fgt_loader, train_retain_loader = get_dsets()
    else:
        all_train_loader,all_test_loader, train_fgt_loader, train_retain_loader, test_fgt_loader, test_retain_loader = get_dsets_remove_class(opt.class_to_be_removed)

    ##### GET MODEL ##### 
    # function to be fixed

    if opt.model== 'resnet18':
        original_pretr_model = get_resnet18_trained()

    elif opt.model== 'resnet50' and opt.dataset == 'VGG':
        original_pretr_model = get_resnet50_trained_on_VGGFace_10_subjects()
    else:
        
        raise NotImplementedError

    
    original_pretr_model.to(opt.device)
    original_pretr_model.eval()
    if opt.run_original:
        print('\n----ORIGINAL MODEL----')
        if not(opt.class_to_be_removed is None):

            get_outputs(test_retain_loader, test_fgt_loader,original_pretr_model,'cifar10_original_model.pkl')

        else:

            get_outputs(test_loader, train_fgt_loader,original_pretr_model,'cifar10_original_model.pkl')

    ##### UNLEARN #####
    pretr_model = deepcopy(original_pretr_model)
    pretr_model.fc = nn.Sequential(nn.Dropout(0.4),pretr_model.fc) 
    pretr_model.to(opt.device)
    pretr_model.eval()

    if opt.run_unlearn:
        print('\n----- UNLEARNED ----') 
        # saves first time checkpoint to compute time interval  
        timestamp1 = time.time()
        if not opt.competitor:
            unlearned_model = unlearning(pretr_model, train_retain_loader, train_fgt_loader,target_accuracy=opt.target_accuracy)
        else:
            approach = choose_competitor(opt.name_competitor)(pretr_model,train_retain_loader, train_fgt_loader)
            unlearned_model = approach.run()
        opt.unlearning_time = time.time() - timestamp1

        if not(opt.class_to_be_removed is None):
            get_outputs(test_retain_loader, test_fgt_loader,unlearned_model,'cifar10_unlearned.pkl')
        else:
            get_outputs(test_loader, train_fgt_loader,unlearned_model,'cifar10_unlearned.pkl')



if __name__ == "__main__":
    main()
