from copy import deepcopy
from dsets import get_dsets_remove_class, get_dsets
import pandas as pd
#to clean up
import random
from utils import accuracy, set_seed, get_retrained_model,get_resnet50_trained_on_VGGFace_10_subjects,get_resnet_trained
import numpy as np
from unlearn import unlearning
from MIA_code.MIA import get_MIA_MLP
from opts import OPT as opt
import torch.nn as nn
from tqdm import tqdm
#from publisher import push_results
import time
from Unlearning_methods import choose_method

def main(all_train_loader,all_test_loader, train_fgt_loader, train_retain_loader, test_fgt_loader, test_retain_loader, class_to_remove):
    original_pretr_model = get_resnet_trained()
    original_pretr_model.to(opt.device)
    original_pretr_model.eval()
    # df_or_model = get_MIA_MLP(train_fgt_loader, test_fgt_loader, original_pretr_model, opt)
    # df_or_model["forget_train_accuracy"] = accuracy(original_pretr_model, train_fgt_loader)
    # df_or_model["retain_train_accuracy"] = accuracy(original_pretr_model, train_retain_loader)
    # df_or_model["forget_test_accuracy"] =  accuracy(original_pretr_model, test_fgt_loader)
    # df_or_model["retain_test_accuracy"] =  accuracy(original_pretr_model, test_retain_loader)


    pretr_model = deepcopy(original_pretr_model)
    pretr_model.to(opt.device)
    pretr_model.eval()
    timestamp1 = time.time()
    
    approach = choose_method(opt.method)(pretr_model,train_retain_loader, train_fgt_loader, class_to_remove)
    unlearned_model = approach.run()
    
    #df_un_model = get_MIA_MLP(train_fgt_loader, test_fgt_loader, unlearned_model, opt)
    
    timest = time.time() - timestamp1

    

    fgt_train = accuracy(unlearned_model, train_fgt_loader)
    rt_train = accuracy(unlearned_model, train_retain_loader)
    fgt_test =  accuracy(unlearned_model, test_fgt_loader)
    rt_test =  accuracy(unlearned_model, test_retain_loader)

    df_un_model = pd.DataFrame(np.array([[timest, fgt_train,rt_train,fgt_test,rt_test]]),columns=['unlearning_time', 
                                              'forget_train_accuracy', 'retain_train_accuracy', 'forget_test_accuracy', 'retain_test_accuracy'])
    
    # rt_model = get_retrained_model()
    # rt_model.to(opt.device)
    # rt_model.eval()
    # df_rt_model = get_MIA_MLP(train_fgt_loader, test_fgt_loader, rt_model, opt)

        
    # df_rt_model["forget_train_accuracy"] = accuracy(rt_model, train_fgt_loader)
    # df_rt_model["retain_train_accuracy"] = accuracy(rt_model, train_retain_loader)
    # df_rt_model["forget_test_accuracy"] =  accuracy(rt_model, test_fgt_loader)
    # df_rt_model["retain_test_accuracy"] =  accuracy(rt_model, test_retain_loader)

    return df_un_model.mean(0)

if __name__ == "__main__":
    df_unlearned_total={1:[], 10:[], 20:[], 30:[], 40:[], 50:[], 60:[], 70:[], 80:[], 90:[], 98:[]}
    df_retained_total=[]
    df_orig_total=[]
    

    for i in range(1):
        set_seed(i)
        classes=[i for i in range(opt.num_classes)]
        random.shuffle(classes)
        for j in [1]+[z for z in range(10,100,10)]+[98]:
            print(f"seed:{i}-classes removed:{len(classes[:j])}")
            opt.class_to_be_removed=classes[:j]
            #opt.RT_model_weights_path=opt.root_folder+f'chks_tiny/best_checkpoint_without_{opt.class_to_be_removed}.pth'
            #print(f'------------{opt.class_to_be_removed}-----------')
            all_train_loader,all_test_loader, train_fgt_loader, train_retain_loader, test_fgt_loader, test_retain_loader = get_dsets_remove_class(classes[:j])
            # row_orig, row_unl, row_ret=main(all_train_loader,all_test_loader, train_fgt_loader, train_retain_loader, test_fgt_loader, test_retain_loader)        
            row_unl=main(all_train_loader,all_test_loader, train_fgt_loader, train_retain_loader, test_fgt_loader, test_retain_loader, classes[:j])
            df_unlearned_total[j].append(row_unl.values)
            break
            # df_retained_total.append(row_ret.values)
    #df_unlearned_total=pd.DataFrame(df_unlearned_total,columns=['unlearning_time', 
    #                                          'forget_train_accuracy', 'retain_train_accuracy', 'forget_test_accuracy', 'retain_test_accuracy'])
    for k,v in df_unlearned_total.items():
        print(f'Classes Removed: {k}')
        df_unlearned_total[k]=pd.DataFrame(v,columns=['unlearning_time', 
                                              'forget_train_accuracy', 'retain_train_accuracy', 'forget_test_accuracy', 'retain_test_accuracy'])
        means = df_unlearned_total[k].mean()
        std_devs = df_unlearned_total[k].std()
        output = "\n".join([f"{col}: {mean:.4f} \\pm {std:.4f}" for col, mean, std in zip(means.index, means, std_devs)])
        print(output)
    # df_retained_total=pd.DataFrame(df_retained_total,columns=['accuracy', 'chance', 'acc | test ex', 'acc | train ex', 'precision', 'recall', 'F1', 
    #                                           'forget_train_accuracy', 'retain_train_accuracy', 'forget_test_accuracy', 'retain_test_accuracy'])
    #df_orig_total=pd.DataFrame(df_orig_total,columns=['accuracy', 'chance', 'acc | test ex', 'acc | train ex', 'precision', 'recall', 'F1', 
    #                                         'forget_train_accuracy', 'retain_train_accuracy', 'forget_test_accuracy', 'retain_test_accuracy'])
    # print(df_orig_total.shape)
    #print('Results retained:\n',df_retained_total.mean(0), '+-', df_retained_total.std(0))
    #push_results(opt, df_orig_total, df_unlearned_total, df_retained_total)

