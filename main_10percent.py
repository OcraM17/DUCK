from copy import deepcopy
from dsets import get_dsets_remove_class, get_dsets
import pandas as pd
#to clean up
from utils import accuracy, set_seed, get_retrained_model,get_resnet50_trained_on_VGGFace_10_subjects,get_resnet18_trained

from unlearn import unlearning
from MIA_code.MIA import get_MIA_MLP
from opts import OPT as opt
import torch.nn as nn
from tqdm import tqdm
from publisher import push_results
import time
from utils import choose_competitor

def main(train_loader, test_loader, train_fgt_loader, train_retain_loader):

    original_pretr_model = get_resnet18_trained()
    original_pretr_model.to(opt.device)
    original_pretr_model.eval()    
    df_or_model = get_MIA_MLP(train_fgt_loader, test_loader, original_pretr_model, opt)
    

    df_or_model["test_accuracy"] = accuracy(original_pretr_model, test_loader)
    df_or_model["forget_accuracy"] = accuracy(original_pretr_model, train_fgt_loader)
    df_or_model["retain_accuracy"] = accuracy(original_pretr_model, train_retain_loader)
    
    print(df_or_model)


    pretr_model = deepcopy(original_pretr_model)
    pretr_model.fc = nn.Sequential(nn.Dropout(0.4),pretr_model.fc) 
    pretr_model.to(opt.device)
    pretr_model.eval()

    timestamp1 = time.time()
    
    if not opt.competitor:
        unlearned_model = unlearning(pretr_model, train_retain_loader, train_fgt_loader,target_accuracy=opt.target_accuracy)
    else:
        approach = choose_competitor(opt.name_competitor)(pretr_model,train_retain_loader, train_fgt_loader,test_loader)
        unlearned_model = approach.run()
    
   
    df_un_model = get_MIA_MLP(train_fgt_loader, test_loader, unlearned_model, opt)
    df_un_model["unlearn_time"] = time.time() - timestamp1

    df_un_model["test_accuracy"] = accuracy(unlearned_model, test_loader)
    df_un_model["forget_accuracy"] = accuracy(unlearned_model, train_fgt_loader)
    df_un_model["retain_accuracy"] = accuracy(unlearned_model, train_retain_loader)
    print(df_un_model)    
    
    return df_or_model.mean(0), df_un_model.mean(0)

if __name__ == "__main__":
    set_seed(opt.seed)
    df_unlearned_total=[]
    df_retained_total=[]
    df_orig_total=[]
    
    seed_list = [0,1,2,3,4,5,6,7,8,42]
    for i in seed_list:
    
        if opt.dataset == "cifar10":
            num=5000
        elif opt.dataset == "cifar100":
            num=5000
        elif opt.dataset == "tinyImagenet":
            num=10000
        file_fgt = f'{opt.root_folder}forget_id_files/forget_idx_{num}_{opt.dataset}_seed_{i}.txt'
        train_loader, test_loader, train_fgt_loader, train_retain_loader = get_dsets(file_fgt=file_fgt)

        row_orig, row_unl=main(train_loader, test_loader, train_fgt_loader, train_retain_loader)

        df_unlearned_total.append(row_unl.values)

        df_orig_total.append(row_orig.values)

    df_orig_total=pd.DataFrame(df_orig_total,columns=['accuracy', 'chance', 'acc | test ex', 'acc | train ex', 'precision', 'recall', 'F1', 
                                             'test_accuracy', 'forget_accuracy', 'retain_accuracy'])
    df_unlearned_total=pd.DataFrame(df_unlearned_total,columns=['accuracy', 'chance', 'acc | test ex', 'acc | train ex', 'precision', 'recall', 'F1', 'unlearning_time',
                                             'test_accuracy', 'forget_accuracy', 'retain_accuracy'])

    
    print(df_orig_total.shape)
    print('Results original:\n',df_orig_total.mean(0))
    print('Results original:\n',df_orig_total.std(0))

    print('Results unlearned:\n',df_unlearned_total.mean(0))
    print('Results unlearned:\n',df_unlearned_total.std(0))

    #push_results(opt, df_orig_total, df_unlearned_total, df_retained_total)

