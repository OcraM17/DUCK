from copy import deepcopy
from dsets import get_dsets_remove_class, get_dsets
import pandas as pd
from error_propagation import Complex
#to clean up
from utils import accuracy, set_seed, get_retrained_model,get_resnet50_trained_on_VGGFace_10_subjects,get_resnet18_trained

from unlearn import unlearning
from MIA_code.MIA import get_MIA_MLP
from opts import OPT as opt
import torch.nn as nn
from tqdm import tqdm
#from publisher import push_results
import time
from utils import choose_competitor
from error_propagation import Complex
def AUS(a_t, a_or, a_f):
    aus=(Complex(1, 0)-(a_or-a_t))/(Complex(1, 0)+abs(a_f-a_t))
    return aus
def main(train_loader, test_loader, train_fgt_loader, train_retain_loader):
    v_orig, v_unlearn, v_rt = None, None, None
    original_pretr_model = get_resnet18_trained()
    original_pretr_model.to(opt.device)
    original_pretr_model.eval()    
    if opt.run_original:
        df_or_model = get_MIA_MLP(train_fgt_loader, test_loader, original_pretr_model, opt)
        

        df_or_model["test_accuracy"] = accuracy(original_pretr_model, test_loader)
        df_or_model["forget_accuracy"] = accuracy(original_pretr_model, train_fgt_loader)
        df_or_model["retain_accuracy"] = accuracy(original_pretr_model, train_retain_loader)
        v_orig= df_or_model.mean(0)
    #print(df_or_model)

    if opt.run_unlearn:
        pretr_model = deepcopy(original_pretr_model)
        pretr_model.fc = nn.Sequential(nn.Dropout(0.4),pretr_model.fc) 
        pretr_model.to(opt.device)
        pretr_model.eval()

        timestamp1 = time.time()
        if not opt.competitor:
            opt.target_accuracy = accuracy(original_pretr_model, test_loader)
            unlearned_model = unlearning(pretr_model, train_retain_loader, train_fgt_loader,target_accuracy=opt.target_accuracy)
            
        else:
            approach = choose_competitor(opt.name_competitor)(pretr_model,train_retain_loader, train_fgt_loader,test_loader)
            unlearned_model = approach.run()
        
    
        df_un_model = get_MIA_MLP(train_fgt_loader, test_loader, unlearned_model, opt)
        df_un_model["unlearn_time"] = time.time() - timestamp1

        df_un_model["test_accuracy"] = accuracy(unlearned_model, test_loader)
        df_un_model["forget_accuracy"] = accuracy(unlearned_model, train_fgt_loader)
        df_un_model["retain_accuracy"] = accuracy(unlearned_model, train_retain_loader)
        v_unlearn=df_un_model.mean(0)

    if opt.run_rt_model:
        rt_model = get_retrained_model()
        rt_model.to(opt.device)
        rt_model.eval()
        df_rt_model = get_MIA_MLP(train_fgt_loader, test_loader, rt_model, opt)

        df_rt_model["test_accuracy"] = accuracy(rt_model, test_loader)
        df_rt_model["forget_accuracy"] = accuracy(rt_model, train_fgt_loader)
        df_rt_model["retain_accuracy"] = accuracy(rt_model, train_retain_loader)
        v_rt=df_rt_model.mean(0)
       
    
    return v_orig, v_unlearn, v_rt

if __name__ == "__main__":
    df_unlearned_total=[]
    df_retained_total=[]
    df_orig_total=[]
    
    seed_list = [0]#,1,2,3,4,5,6,7,8,42]
    for i in seed_list:
        set_seed(i)

        print(f"Seed {i}")
    
        if opt.dataset == "cifar10":
            num=5000
        elif opt.dataset == "cifar100":
            num=5000
        elif opt.dataset == "tinyImagenet":
            num=10000
        file_fgt = f'{opt.root_folder}forget_id_files/forget_idx_{num}_{opt.dataset}_seed_{i}.txt'
        train_loader, test_loader, train_fgt_loader, train_retain_loader = get_dsets(file_fgt=file_fgt)
        opt.RT_model_weights_path=opt.root_folder+f'chks_{opt.dataset if opt.dataset!="tinyImagenet" else "tiny"}/chks_{opt.dataset if opt.dataset!="tinyImagenet" else "tiny"}_seed_{i}.pth'
        print(opt.RT_model_weights_path)

        row_orig, row_unl, row_ret=main(train_loader, test_loader, train_fgt_loader, train_retain_loader)
        if row_unl is not None:
            df_unlearned_total.append(row_unl.values)
        if row_orig is not None:
            df_orig_total.append(row_orig.values)
        if row_ret is not None:
            df_retained_total.append(row_ret.values)
        
    print(opt.dataset)
    if df_orig_total:
        print("ORIGINAL \n")
        df_orig_total=pd.DataFrame(df_orig_total,columns=['accuracy', 'chance', 'acc | test ex', 'acc | train ex', 'precision', 'recall', 'F1','mutual' ,
                                             'test_accuracy', 'forget_accuracy', 'retain_accuracy'])
        #print(df_orig_total.shape)
        #print('Results original:\n',df_orig_total.mean(0))
        #print('Results original:\n',df_orig_total.std(0))
        means = df_orig_total.mean()
        std_devs = df_orig_total.std()
        output = "\n".join([f"{col}: {mean*100:.4f} \\pm {std*100:.4f}" for col, mean, std in zip(means.index, means, std_devs)])
        print(output)

    if df_unlearned_total:
        print("UNLEARNED \n")
        df_unlearned_total=pd.DataFrame(df_unlearned_total,columns=['accuracy', 'chance', 'acc | test ex', 'acc | train ex', 'precision', 'recall', 'F1','mutual', 'unlearning_time',
                                             'test_accuracy', 'forget_accuracy', 'retain_accuracy'])
        #print('Results unlearned:\n',df_unlearned_total.mean(0))
        #print('Results unlearned:\n',df_unlearned_total.std(0))
        means = df_unlearned_total.mean()
        std_devs = df_unlearned_total.std()
        output = "\n".join([f"{col}: {100*mean:.2f} \\pm {100*std:.2f}" if col != 'unlearning_time' else f"{col}: {mean:.2f} \\pm {std:.2f}" for col, mean, std in zip(means.index, means, std_devs)])
        
        print(output)
        a_t=Complex(means["test_accuracy"], std_devs["test_accuracy"])
        a_or = opt.a_or[opt.dataset]
        a_r = Complex(means["forget_accuracy"], std_devs["forget_accuracy"])
        aus = AUS(a_t, a_or, a_r)
        print(f"AUS: {aus.value:.4f} \pm {aus.error:.4f}")
        
        #save df_unlearned_total
        df_unlearned_total.to_csv(f"{opt.root_folder}unlearned_results_{opt.dataset if opt.dataset!='tinyImagenet' else 'tiny'}.csv")


    if df_retained_total:
        print("RETAINED \n")
        df_retained_total=pd.DataFrame(df_retained_total,columns=['accuracy', 'chance', 'acc | test ex', 'acc | train ex', 'precision', 'recall', 'F1','mutual','test_accuracy', 'forget_accuracy', 'retain_accuracy'])
        #print('Results retained:\n',df_retained_total.mean(0))
        #print('Results retained:\n',df_retained_total.std(0))
        means = df_retained_total.mean()
        std_devs = df_retained_total.std()
        output = "\n".join([f"{col}: {100*mean:.2f} \\pm {100*std:.2f}" for col, mean, std in zip(means.index, means, std_devs)])
        print(output)

    #push_results(opt, df_orig_total, df_unlearned_total, df_retained_total)

