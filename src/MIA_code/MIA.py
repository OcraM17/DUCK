import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils
from torch.utils.data import DataLoader
from torchvision import transforms
from opts import OPT as opt
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve,auc
import warnings
import os
import glob
from copy import deepcopy
import math
from tqdm import tqdm
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

import pandas as pd

def compute_f1_score(predicted_labels, true_labels):
    """
    Compute the F1 score given the predicted labels and true labels.

    Args:
        predicted_labels (list or array-like): The predicted labels.
        true_labels (list or array-like): The true labels.

    Returns:
        float: The F1 score.
    """
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for predicted, true in zip(predicted_labels, true_labels):
        if predicted == 1 and true == 1:
            true_positives += 1
        elif predicted == 1 and true == 0:
            false_positives += 1
        elif predicted == 0 and true == 1:
            false_negatives += 1

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)

    f1_score = 2 * (precision * recall) / (precision + recall)
    return precision,recall,f1_score

#function to compute accuracy given a model and a dataloader
def compute_accuracy(model, dataloader,device,targ_val = None):
    model.eval()
    correct = 0
    total = 0 
    prediction_F1 = []
    labels = []
    for inputs, targets in dataloader:
        if targ_val is None:
            inputs, targets = inputs.to(device), targets.to(device)
        else:
            inputs, targets = inputs[targets==targ_val].to(device), targets[targets==targ_val].to(device)

        outputs = model(inputs)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        prediction_F1.extend(predicted.detach().cpu().numpy())
        labels.extend(targets.detach().cpu().numpy())
    #compute F1 score
    if targ_val is None:
        labels= np.asarray(labels)
        chance = labels.sum()/labels.shape[0]
        P,R,F1 = precision_recall_fscore_support(np.asarray(labels).astype(np.int64), np.asarray(prediction_F1).astype(np.int64), average='binary')[:3]
        return correct / total, chance ,P,R,F1
    else:
        return correct / total
def compute_mutual_information(vector1, vector2):
    import math
    """
    Compute the mutual information between two binary vectors.

    Args:
        vector1 (list or array-like): The first binary vector.
        vector2 (list or array-like): The second binary vector.

    Returns:
        float: The mutual information.
    """
    assert len(vector1) == len(vector2), "Vectors must have the same length."
    
    n = len(vector1)
    
    # Calculate the joint probability distribution
    joint_prob = [[0, 0], [0, 0]]
    
    for i in range(n):
        joint_prob[vector1[i]][vector2[i]] += 1
    
    joint_prob = [[count / n for count in row] for row in joint_prob]
    print(joint_prob)
    # Calculate the marginal probability distributions
    marg_prob1 = [sum(joint_prob[i][j] for j in range(2)) for i in range(2)]
    marg_prob2 = [sum(joint_prob[i][j] for i in range(2)) for j in range(2)]
    
    # Calculate the mutual information
    mutual_info = 0
    
    for i in range(2):
        for j in range(2):
            if joint_prob[i][j] > 0 and marg_prob1[i] > 0 and marg_prob2[j] > 0:
                mutual_info += joint_prob[i][j] * \
                    (math.log(joint_prob[i][j] /
                              (marg_prob1[i] * marg_prob2[j]), 2))
    
    return mutual_info
def compute_accuracy_SVC(predicted, labels, targ_val = None):
    if targ_val is not None:
        predicted, labels = predicted[labels==targ_val], labels[labels==targ_val]

    total = len(predicted)
    correct = np.equal(predicted,labels).sum().item()
    #compute F1 score
    if targ_val is None:
        labels= np.asarray(labels)
        chance = labels.sum()/labels.shape[0]
        P,R,F1 = precision_recall_fscore_support(np.asarray(labels).astype(np.int64), np.asarray(predicted).astype(np.int64), average='micro')[:3]
        mutual = compute_mutual_information(np.asarray(labels).astype(np.int64), np.asarray(predicted).astype(np.int64))
        return correct / total, chance ,P,R,F1, mutual
    else:
        return correct / total

def training_SVC(model,X_train, X_test, z_train, z_test,opt):
    param_grid = {'C': [1,5,10,100],
              'gamma': [10,1, 0.1], 
              'kernel': ['rbf']}
    grid = GridSearchCV(model, param_grid, refit = True, verbose=3 if opt.verboseMIA else 0, cv=3, n_jobs=4) 
    grid.fit(X_train, z_train)
    best_model = grid.best_estimator_
    print(grid.best_params_)

    results = best_model.predict(X_test)

    accuracy,chance, precision, recall,F1, mutual = compute_accuracy_SVC(results, z_test)
    accuracy_test_ex =0# compute_accuracy_SVC(results, z_test,0)
    accuracy_train_ex =0# compute_accuracy_SVC(results, z_test,1)

    if opt.verboseMIA:
        print(f'Test accuracy: {round(accuracy,3)}')
        #print accuracy for test set with targets equal to 0 or 1   
        print(f'Test accuracy for case test examples: {round(accuracy_test_ex,3)}')
        print(f'Test accuracy for case training examples: {round(accuracy_train_ex,3)}')
    print(f'Test F1: {round(F1,3)}')
    return accuracy,chance,accuracy_test_ex,accuracy_train_ex, precision, recall,F1, mutual

def expand_data(data):
    data1 = torch.zeros_like(data)
    data2 = torch.zeros_like(data)
    data3 = torch.zeros_like(data)
    tranf1 = transforms.RandomCrop(32, padding=4)
    transf2 = transforms.Compose([transforms.RandomHorizontalFlip()])
    #transf3 = transforms.RandomRotation(degrees=(-50, 50))
    for i in range(data.shape[0]):
        data1[i] = tranf1(data[i])
        data2[i] = transf2(data[i])
        #data3[i] = transf3(data[i])
    data = torch.cat((data,data1,data2),dim=0)
    return data

def collect_prob(data_loader, model,opt,exp=False):
    #bbone = torch.nn.Sequential(*(list(self.net.children())[:-1] + [nn.Flatten()]))
    #data_loader = torch.utils.data.DataLoader(data_loader.dataset, batch_size=256, shuffle=True)
    prob = []

    with torch.no_grad():
        for idx, batch in enumerate(data_loader):
            data, target = batch
            if exp:
                data = expand_data(data)
            output = model(data.to(opt.device))
            loss = F.cross_entropy(output, target.to(opt.device),reduce=False)
            prob.append(F.cross_entropy(output, target.to(opt.device),reduce=False)[:,None].cpu())#F.softmax(output, dim=1).detach().cpu())#
        prob=torch.cat(prob)       
        # for i in range(prob.shape[0]):
        #     prob[i] = torch.roll(prob[i],np.random.randint(0,prob.shape[1]),dims=0)         
    return prob

def get_membership_attack_data(train_loader, test_loader, model,opt):    
    #get membership attack data, this function will return X_r, Y_r, X_f, Y_f
    
    train_prob = collect_prob(train_loader, model,opt)
    test_prob = collect_prob(test_loader, model,opt)

    X_tr = train_prob.cpu()
    Y_tr = torch.zeros(len(train_prob),dtype=torch.int64)
    
    X_te = test_prob.cpu()
    Y_te = torch.ones(len(test_prob),dtype=torch.int64)


    N_tr = X_tr.shape[0]
    N_te = X_te.shape[0]

    #sample from training data N_r samples
    Idx = np.arange(N_tr)
    np.random.shuffle(Idx)
    X_tr = X_tr[Idx[:N_te],:]
    Y_tr = Y_tr[Idx[:N_te]]
    N_tr = X_tr.shape[0]

    xtrain = torch.cat([X_tr[:int(0.8*N_tr)],X_te[:int(0.8*N_te)]],dim=0)
    ytrain = torch.cat([Y_tr[:int(0.8*N_tr)],Y_te[:int(0.8*N_te)]],dim=0)

    xtest = torch.cat([X_tr[int(0.8*N_tr):],X_te[int(0.8*N_te):]],dim=0) 
    ytest = torch.cat([Y_tr[int(0.8*N_tr):],Y_te[int(0.8*N_te):]],dim=0)

    # Compute entropy
    entropy = torch.sum(-train_prob*torch.log(torch.clamp(train_prob,min=1e-5)),dim=1)
    train_entropy = torch.mean(entropy).item()
    train_entropy_std = torch.std(entropy).item()
    print(f"train entropy: {train_entropy} +- {train_entropy_std}")
    entropy=torch.sum(-test_prob*torch.log(torch.clamp(test_prob,min=1e-5)),dim=1)
    test_entropy = torch.mean(entropy).item()
    test_entropy_std = torch.std(entropy).item()
    print(f"test entropy: {test_entropy} +- {test_entropy_std}")

    if opt.verboseMIA: 
        print(f'Train and test classification chance, train: {ytrain.sum()/ytrain.shape[0]}, chance test {ytest.sum()/ytest.shape[0]}')
        print('check input vectors: ',torch.unique(ytrain),torch.unique(ytest),torch.max(xtrain),torch.max(xtest))
    return xtrain,ytrain,xtest,ytest,train_entropy, test_entropy

# def get_MIA_SVC(train_loader, test_loader, model, opt):
#     results = []
#     for i in range(opt.iter_MIA):
#         train_data, train_labels, test_data, test_labels, train_entropy, test_entropy = get_membership_attack_data(train_loader, test_loader, model, opt)
        
#         model_SVC = SVC( tol = 1e-4, max_iter=4000, class_weight='balanced', random_state=i) 
#         accuracy, chance,accuracy_test_ex,accuracy_train_ex, P,R,F1, mutual = training_SVC(model_SVC, train_data, test_data, train_labels, test_labels, opt)
#         results.append(np.asarray([accuracy, chance,accuracy_test_ex,accuracy_train_ex,P,R,F1, mutual,train_entropy, test_entropy]))
    
#     results = np.asarray(results)
#     df = pd.DataFrame(results,columns=['accuracy','chance','acc | test ex','acc | train ex','precision','recall','F1', 'Mutual', "Train Entropy", "Test Entropy"])
#     return df

mean = {
        'cifar10': (0.4914, 0.4822, 0.4465),
        'cifar100': (0.5071, 0.4867, 0.4408),
        'tinyImagenet': (0.485, 0.456, 0.406),
        }

std = {
        'cifar10': (0.2023, 0.1994, 0.2010),
        'cifar100': (0.2675, 0.2565, 0.2761),
        'tinyImagenet': (0.229, 0.224, 0.225),
        }

list_test = [
        transforms.ToTensor(),
        transforms.Normalize(mean[opt.dataset],std[opt.dataset]),
    ]
transform_test= transforms.Compose(list_test)



# transform_dset = transforms.Compose(
#         [   transforms.RandomCrop(64, padding=8) if opt.dataset == 'tinyImagenet' else transforms.RandomCrop(32, padding=4),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             transforms.Normalize(mean[opt.dataset],std[opt.dataset]),
#         ]
#     )
def get_membership_attack_data_CR(train_loader, test_loader,model,opt,fgt_loader=None,fgt_loader_t=None):    
    #get membership attack data, this function will return X_r, Y_r, X_f, Y_f
    train_loader.dataset.transform = transform_test
    fgt_loader.dataset.transform = transform_test
    #print(fgt_loader.dataset.transform)
    #### fgt test to expand

    # train_prob = collect_prob(train_loader, model,opt,exp=False)
    # Y = torch.zeros(len(train_prob),dtype=torch.int64)

    # test_prob = collect_prob(test_loader, model,opt,exp=False)
    # Y_test = torch.ones(len(test_prob),dtype=torch.int64)

    fgt_prob = collect_prob(fgt_loader, model,opt,exp=False)
    Y_fgt = torch.zeros(len(fgt_prob),dtype=torch.int64)
    
    fgt_prob_t = collect_prob(fgt_loader_t, model,opt,exp=False)
    Y_fgt_t = torch.ones(len(fgt_prob_t),dtype=torch.int64)

    # N_test = 2000#test_prob.shape[0]

    # X = train_prob.cpu()
    # X_test= test_prob.cpu()

    # X = X[:N_test,:]
    # Y = Y[:N_test]

    # X_test = X_test[:N_test,:]
    # Y_test = Y_test[:N_test]



    # xtrain = torch.cat([X,X_test],dim=0)
    # ytrain = torch.cat([Y,Y_test],dim=0)

    # xtest = fgt_prob.cpu()#torch.cat([X_fgt_t[n_sampt:,:],X_fgt[n_samp:,:]],dim=0)
    # ytest = Y_fgt#torch.cat([Y_fgt_t[n_sampt:],Y_fgt[n_samp:]],dim=0)

    # #set exp to True in Tiny
    # fgt_prob_t = collect_prob(fgt_loader_t, model,opt,exp=False)
    # print(torch.unique(torch.argmax(fgt_prob_t,dim=1),return_counts=True))

    X_fgt = fgt_prob.cpu()
    # Y_fgt = torch.zeros(len(fgt_prob),dtype=torch.int64)

    X_fgt_t = fgt_prob_t.cpu()

    import matplotlib.pyplot as plt
    
    # Y_fgt_t = torch.ones(len(fgt_prob_t),dtype=torch.int64)
    print(X_fgt.mean(),X_fgt.std(),X_fgt_t.mean(),X_fgt_t.std())
    #entropy=torch.sum(-fgt_prob*torch.log(torch.clamp(fgt_prob,min=1e-5)),dim=1)
      
    #plt.hist(X_fgt[:,0].numpy().flatten(),density=True,bins=100,alpha=.3,color='blue')
    #X_fgt = entropy[:,None].cpu()
    # fgt_entropy = torch.mean(entropy).item()
    # fgt_entropy_std = torch.std(entropy).item()
    # print(f"fgt entropy: {fgt_entropy} +- {fgt_entropy_std}")


    #entropy=torch.sum(-fgt_prob_t*torch.log(torch.clamp(fgt_prob_t,min=1e-5)),dim=1)
    #plt.hist(X_fgt_t[:,0].numpy().flatten(),density=True,bins=100,alpha=.3,color='red')
    #plt.savefig('test_ent.png')
    #input('ccc')
    # #X_fgt_t = entropy[:,None].cpu()
    # fgt_entropy_t = torch.mean(entropy).item()
    # fgt_entropy_std = torch.std(entropy).item()
    # print(f"fgt_t entropy: {fgt_entropy_t} +- {fgt_entropy_std}")

    N_fgt_t = X_fgt_t.shape[0]
    N_fgt = X_fgt.shape[0]
    X_fgt_t = X_fgt_t[torch.randperm(N_fgt_t),:]
    X_fgt = X_fgt[torch.randperm(N_fgt),:]

  

    X_fgt = X_fgt[:N_fgt_t,:]
    Y_fgt = Y_fgt[:N_fgt_t]
    N_fgt = X_fgt.shape[0]


    n_samp = int(0.7*N_fgt) 
    n_sampt = int(0.7*N_fgt_t)

    xtrain = torch.cat([X_fgt_t[:n_sampt,:],X_fgt[:n_samp,:]],dim=0)
    ytrain = torch.cat([Y_fgt_t[:n_sampt],Y_fgt[:n_samp]],dim=0)

    xtest = torch.cat([X_fgt_t[n_sampt:,:],X_fgt[n_samp:,:]],dim=0)
    ytest = torch.cat([Y_fgt_t[n_sampt:],Y_fgt[n_samp:]],dim=0)

    # N_fgt_t = X_fgt_t.shape[0]
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    xtrain = scaler.fit_transform(xtrain)
    xtest = scaler.transform(xtest)
    # np.random.shuffle(X_fgt_t)
    # np.random.shuffle(X_fgt)
    # X_fgt = X_fgt[:N_fgt_t*3,:]
    # Y_fgt = Y_fgt[:N_fgt_t*3]
    # N_fgt = X_fgt.shape[0]


    # n_samp = int(0.7*N_fgt) 
    # n_sampt = int(0.7*N_fgt_t)

    # xtrain = torch.cat([X_fgt_t[:n_sampt,:],X_fgt[:n_samp,:]],dim=0)
    # ytrain = torch.cat([Y_fgt_t[:n_sampt],Y_fgt[:n_samp]],dim=0)

    # xtest = torch.cat([X_fgt_t[n_sampt:,:],X_fgt[n_samp:,:]],dim=0)
    # ytest = torch.cat([Y_fgt_t[n_sampt:],Y_fgt[n_samp:]],dim=0)


    # Compute entropy



    print('check input vectors: ',xtrain.shape,xtest.shape,np.unique(ytrain,return_counts=True))
    return xtrain,ytrain,xtest,ytest

def get_MIA_SVC(train_loader, test_loader, model, opt,fgt_loader=None,fgt_loader_t=None):
    results = []
    for i in range(opt.iter_MIA):
        if opt.mode == "HR":
            train_data, train_labels, test_data, test_labels, train_entropy, test_entropy = get_membership_attack_data(train_loader, test_loader, model, opt)
        elif opt.mode == "CR":
            train_data, train_labels, test_data, test_labels = get_membership_attack_data_CR(train_loader, test_loader, model, opt,fgt_loader=fgt_loader, fgt_loader_t=fgt_loader_t)
        model_SVC = SVC( tol = 1e-4, max_iter=4000, random_state=i,class_weight='balanced')
        accuracy, chance,accuracy_test_ex,accuracy_train_ex, P,R,F1, mutual = training_SVC(model_SVC, train_data, test_data, train_labels, test_labels, opt)
        results.append(np.asarray([accuracy, chance,accuracy_test_ex,accuracy_train_ex,P,R,F1, mutual,0, 0]))
    
    results = np.asarray(results)
    df = pd.DataFrame(results,columns=['accuracy','chance','acc | test ex','acc | train ex','precision','recall','F1', 'Mutual', "Train Entropy", "Test Entropy"])
    return df

##############################################################################################################################

def comp_confidence_stable(output,target):
    confidence = torch.exp(-F.cross_entropy(output, target.to(opt.device),reduce=False)[:,None].cpu())

    confidence_out = []
    for i in range(opt.num_classes):
        
        t2 = torch.ones_like(target,dtype=target.dtype)
        t2=t2*i
        confidence_i = torch.exp(-F.cross_entropy(output, t2.to(opt.device),reduce=False)[:,None].cpu())
        confidence_i[target==i]=0
        confidence_out.append(confidence_i)
    confidence_out = torch.cat(confidence_out,dim=1)
    #print(confidence.shape,torch.log(torch.sum(confidence_out,dim=1)).shape)
    return torch.log(confidence[0]+0.000001) - torch.log(torch.sum(confidence_out,dim=1))

from utils import accuracy
def collect_prob_logits(data_loader, model,opt):

    prob_all = []
    #print('acc',accuracy(model,data_loader))
    transformation = [None]#,
                    #   transforms.RandomHorizontalFlip(),
                    # transforms.RandomCrop(32, padding=4),
                    # transforms.RandomRotation(15)]
    model.eval()
    with torch.no_grad():
        for transf in transformation:
            prob=[]
            for idx, batch in enumerate(data_loader):

                data, target = batch
                if transf is None:
                    data = data
                else:
                    data = transf(data)
                output = model(data.to(opt.device))
                #unstable version
                confidence = torch.exp(-F.cross_entropy(output, target.to(opt.device),reduce=False)[:,None].cpu())
                buff = 0.0000001+(confidence/(1-confidence + 0.0000001))
                prob.append(torch.log(buff))
                # if idx==0:
                #     print(confidence.mean())
                #stable version
                #prob.append(comp_confidence_stable(output,target))
                
                #print((torch.argmax(output,dim=1)==target.to(opt.device)).sum()/len(target.to(opt.device)))
            prob=torch.cat(prob)    
            prob_all.append(prob)  
        # print(prob.max(),prob.min())
    prob_all = torch.cat(prob_all,dim=1)

    return prob_all[:,None,:]

def get_MIA(train_fgt_loader, test_loader, model,opt,original_pretr_model):
    
    #compute for fgt samples distributions from shadow models
    if opt.mode == "HR":
        weights = glob.glob(f'{opt.root_folder}/weights_shadow/chks_{opt.dataset}/{opt.mode}/seed_{opt.seed}/*.pth')
    else:
        weights = glob.glob(f'{opt.root_folder}/weights_shadow/chks_{opt.dataset}/{opt.mode}/class_{opt.class_to_remove[0][0]}/*.pth')
        #load idx for forget set and remove them from fgt set 
        indices_fgt = np.loadtxt(f'{opt.root_folder}forget_id_files_shadow/forget_idx_{opt.dataset}_seed_{opt.seed[0]}_class_{opt.class_to_remove[0][0]}.txt').astype(np.int64)
        
        mask = np.zeros(train_fgt_loader.dataset.indices.shape, dtype=bool)
        mask[indices_fgt] = True
        idx_samples = np.arange(mask.size)[~mask]
        half_fgt_set = torch.utils.data.Subset(train_fgt_loader.dataset, idx_samples)#idx_samples)
        train_fgt_loader = torch.utils.data.DataLoader(half_fgt_set, batch_size=opt.batch_size,shuffle=True, num_workers=opt.num_workers)

    prob_test_model_test = collect_prob_logits(test_loader, model,opt)
    prob_test_model = collect_prob_logits(train_fgt_loader, model,opt)
    print(prob_test_model.shape)
    print(f'got {len(weights)} shadow models...')
    shadow_model = deepcopy(original_pretr_model)
    shadow_model.eval()
    model.eval()

    distributions = []
    distributions_test = []
    weights.sort()
    for weights_name in tqdm(weights[:]):
        #try:
        shadow_model.load_state_dict(torch.load(weights_name))
        shadow_model.eval()
        prob = collect_prob_logits(train_fgt_loader, shadow_model,opt)
        prob_test = collect_prob_logits(test_loader, shadow_model,opt)
        
        distributions.append(prob)
        distributions_test.append(prob_test)
        # except:
        #    print('error in loading model')
        #    pass

    distributions = torch.cat(distributions,dim=1)
    distributions_test = torch.cat(distributions_test,dim=1)

    #make 10 subplots 2x5
    # fig, axs = plt.subplots(4,5,figsize=(15,8))
    # for i in range(20):
        
    #     axs[i//5,i%5].hist(distributions[i,:].cpu().numpy(),bins=20,color='orange',alpha=0.5)
    #     #axs[i//5,i%5].hist(distributions_test[i,:].cpu().numpy(),bins=20,color='blue',alpha=0.5)
    #     #plot a vertical line at 0.5
    #     axs[i//5,i%5].axvline(x=prob_test_model[i,0].cpu().numpy(),color='red',linewidth=2)
    #     #axs[i//5,i%5].axvline(x=prob_test_model_test[i,0].cpu().numpy(),color='k',linewidth=2)

    # plt.tight_layout()
    # plt.savefig('hist25.png')
    # plt.close()
    #compute mu and std for each fgt sample
    mu = torch.mean(distributions,dim=1)
    std = torch.std(distributions,dim=1)
    mu_test = torch.mean(distributions_test,dim=1)
    std_test = torch.std(distributions_test,dim=1)
    print(mu.shape)
    for i in range(1):
        if i==0:
            final_prob_vector = 0.5*(1-torch.special.erf((prob_test_model[:,0,i]-mu[:,i])/(math.sqrt(2)*std[:,i])))
            final_prob_vector_test = 0.5*(1-torch.special.erf((prob_test_model_test[:,0,i]-mu_test[:,i])/(math.sqrt(2)*std_test[:,i])))
            buff = prob_test_model_test[:,0,i].cpu().numpy()
            buff2 = mu_test[:,i].cpu().numpy()+std_test[:,i].cpu().numpy()
            print('NUM: ',buff2[buff>buff2].shape)
            plt.scatter(buff2[buff>buff2],buff[buff>buff2])
            plt.ylim([-4,4])
            plt.xlim([-4,4])
            plt.savefig('check.png')
            plt.close()
        else:
            final_prob_vector = final_prob_vector*0.5*(1-torch.special.erf((prob_test_model[:,0,i]-mu[:,i])/(math.sqrt(2)*std[:,i])))
            final_prob_vector_test = final_prob_vector_test*0.5*(1-torch.special.erf((prob_test_model_test[:,0,i]-mu_test[:,i])/(math.sqrt(2)*std_test[:,i])))
    
    print('Mean std model fgt: ',(1-final_prob_vector).mean(),(1-final_prob_vector).std())
    print('Mean std model fgt test:',(1-final_prob_vector_test).mean(),(1-final_prob_vector_test).std())
    y_lab = np.concatenate((np.ones_like(final_prob_vector.numpy()), np.zeros_like(final_prob_vector_test.numpy())),axis=0)
    y_pred = np.concatenate((1-final_prob_vector.numpy(), 1-final_prob_vector_test.numpy()),axis=0)
    fpr, tpr, thresholds = roc_curve(y_lab,y_pred)

    #SAVE FPR, TPR AND THRESHOLDS as a .npy file
    print(f'{opt.root_folder}/results')
    os.makedirs(f'{opt.root_folder}/results', exist_ok=True)
    if opt.mode=='HR':
        if opt.run_unlearn:
            path = f'{opt.root_folder}/results/fpr_tpr_{opt.dataset}_{opt.mode}_{opt.seed[0]}_unlearned.npy' 
        elif opt.run_original:
            path = f'{opt.root_folder}/results/fpr_tpr_{opt.dataset}_{opt.mode}_{opt.seed[0]}_original.npy' 
        elif opt.run_rt_model:
            path = f'{opt.root_folder}/results/fpr_tpr_{opt.dataset}_{opt.mode}_{opt.seed[0]}_retrained.npy' 
    else:
        if opt.run_unlearn:
            path = f'{opt.root_folder}/results/fpr_tpr_{opt.dataset}_{opt.mode}_{opt.seed[0]}_class_{opt.class_to_remove[0][0]}_unlearned.npy' 
        elif opt.run_original:
            path = f'{opt.root_folder}/results/fpr_tpr_{opt.dataset}_{opt.mode}_{opt.seed[0]}_class_{opt.class_to_remove[0][0]}_original.npy' 
        elif opt.run_rt_model:
            path = f'{opt.root_folder}/results/fpr_tpr_{opt.dataset}_{opt.mode}_{opt.seed[0]}_class_{opt.class_to_remove[0][0]}_retrained.npy' 

    np.save(path,np.asarray([fpr,tpr,thresholds]))


    #plot fpr e tpr distributions

    plt.hist(1-final_prob_vector.numpy(),bins=100,color='blue',alpha=0.5,label='fgt',density=True)
    plt.hist(1-final_prob_vector_test.numpy(),bins=100,color='red',alpha=0.5,label='test_fgt',density=True)
    #plt.xscale('log')
    plt.xlim([0.8,1])
    plt.legend()
    plt.savefig('histogram_mia_distrib.png')
    #plot histogram of final_prob_vector
    # import matplotlib.pyplot as plt
    # plt.hist(final_prob_vector.numpy(),bins=100)
    # plt.xlabel('Prob')
    # plt.ylabel('Count')
    # plt.savefig("histogram.png")
    #plot roc curve
    # import matplotlib.pyplot as plt
    # print(fpr,tpr, thresholds)
    # plt.plot(fpr, tpr)
    # plt.plot([10**-3,1],[10**-3,1], ls="--", c=".3")
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.xlim([10**-3, 1.0])
    # plt.ylim([10**-3, 1.0])
    
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('ROC')
    # #use tight_layout
    # plt.tight_layout()
    # plt.savefig("ROC.png")
    print(f'AUC: {auc(fpr, tpr)}')
    return final_prob_vector.mean()

    # fgt train: (0.9976, tensor(0.8140)), fgt test: (0.8556, tensor(0.6795)),fgt test nel test: (0.87, tensor(0.6987))
    # Epoch: 84, Train Loss: 0.892, Train Acc: 99.156, Val Acc: 83.630, Best Acc: 83.770

    # fgt train: (0.9956, tensor(0.8132)), fgt test: (0.848, tensor(0.6755)),fgt test nel test: (0.865, tensor(0.6952))
    # Epoch: 84, Train Loss: 0.892, Train Acc: 99.136, Val Acc: 83.710, Best Acc: 83.720