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

def main(all_train_loader,all_test_loader, train_fgt_loader, train_retain_loader, test_fgt_loader, test_retain_loader):
    original_pretr_model = get_resnet18_trained()
    original_pretr_model.to(opt.device)
    original_pretr_model.eval()
    df_or_model = get_MIA_MLP(train_fgt_loader, test_fgt_loader, original_pretr_model, opt)
    df_or_model["forget_train_accuracy"] = accuracy(original_pretr_model, train_fgt_loader)
    df_or_model["retain_train_accuracy"] = accuracy(original_pretr_model, train_retain_loader)
    df_or_model["forget_test_accuracy"] =  accuracy(original_pretr_model, test_fgt_loader)
    df_or_model["retain_test_accuracy"] =  accuracy(original_pretr_model, test_retain_loader)


    pretr_model = deepcopy(original_pretr_model)
    pretr_model.fc = nn.Sequential(nn.Dropout(0.4),pretr_model.fc) 
    pretr_model.to(opt.device)
    pretr_model.eval()
    timestamp1 = time.time()
    
    if not opt.competitor:
        unlearned_model = unlearning(pretr_model, train_retain_loader, train_fgt_loader,target_accuracy=opt.target_accuracy)
    else:
        approach = choose_competitor(opt.name_competitor)(pretr_model,train_retain_loader, train_fgt_loader)
        unlearned_model = approach.run()
    
    
    df_un_model = get_MIA_MLP(train_fgt_loader, test_fgt_loader, unlearned_model, opt)
    df_un_model["unlearn_time"] = time.time() - timestamp1

    df_un_model["forget_train_accuracy"] = accuracy(unlearned_model, train_fgt_loader)
    df_un_model["retain_train_accuracy"] = accuracy(unlearned_model, train_retain_loader)
    df_un_model["forget_test_accuracy"] =  accuracy(unlearned_model, test_fgt_loader)
    df_un_model["retain_test_accuracy"] =  accuracy(unlearned_model, test_retain_loader)
    
    rt_model = get_retrained_model()
    rt_model.to(opt.device)
    rt_model.eval()
    df_rt_model = get_MIA_MLP(train_fgt_loader, test_fgt_loader, rt_model, opt)

        
    df_rt_model["forget_train_accuracy"] = accuracy(rt_model, train_fgt_loader)
    df_rt_model["retain_train_accuracy"] = accuracy(rt_model, train_retain_loader)
    df_rt_model["forget_test_accuracy"] =  accuracy(rt_model, test_fgt_loader)
    df_rt_model["retain_test_accuracy"] =  accuracy(rt_model, test_retain_loader)
    
    return df_or_model.mean(0), df_un_model.mean(0), df_rt_model.mean(0)

if __name__ == "__main__":
    set_seed(opt.seed)
    df_unlearned_total=[]
    df_retained_total=[]
    df_orig_total=[]
    

    for i in range(2):
        opt.class_to_be_removed=i*10
        print(f'------------{opt.class_to_be_removed}-----------')
        all_train_loader,all_test_loader, train_fgt_loader, train_retain_loader, test_fgt_loader, test_retain_loader = get_dsets_remove_class(opt.class_to_be_removed)
        row_orig, row_unl, row_ret=main(all_train_loader,all_test_loader, train_fgt_loader, train_retain_loader, test_fgt_loader, test_retain_loader)
        df_unlearned_total.append(row_unl.values)
        df_retained_total.append(row_ret.values)
        df_orig_total.append(row_orig.values)
    df_unlearned_total=pd.DataFrame(df_unlearned_total,columns=['accuracy', 'chance', 'acc | test ex', 'acc | train ex', 'precision', 'recall', 'F1', 'unlearning_time', 
                                             'forget_train_accuracy', 'retain_train_accuracy', 'forget_test_accuracy', 'retain_test_accuracy'])
    df_retained_total=pd.DataFrame(df_retained_total,columns=['accuracy', 'chance', 'acc | test ex', 'acc | train ex', 'precision', 'recall', 'F1', 
                                             'forget_train_accuracy', 'retain_train_accuracy', 'forget_test_accuracy', 'retain_test_accuracy'])
    df_orig_total=pd.DataFrame(df_orig_total,columns=['accuracy', 'chance', 'acc | test ex', 'acc | train ex', 'precision', 'recall', 'F1', 
                                             'forget_train_accuracy', 'retain_train_accuracy', 'forget_test_accuracy', 'retain_test_accuracy'])
    print(df_orig_total.shape)
    print('Results original:\n',df_orig_total.mean(0))
    print('Results unlearned:\n',df_unlearned_total.mean(0))
    print('Results retained:\n',df_retained_total.mean(0))
    push_results(opt, df_orig_total, df_unlearned_total, df_retained_total)

