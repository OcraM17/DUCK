import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.svm import SVC

import pandas as pd

# deep mlp
class DeepMLP(nn.Module):
    def __init__(self,input_classes=10, num_classes=2, num_layers=3, num_hidden=100):
        super(DeepMLP, self).__init__()
        self.num_layers = num_layers
        self.num_hidden = num_hidden

        self.fc1 = nn.Linear(input_classes, num_hidden)
        self.bn1 = nn.BatchNorm1d(num_hidden)
        self.fully_connected = nn.ModuleList([nn.Linear(num_hidden, num_hidden) for _ in range(num_layers)])
        self.fc = nn.Linear(num_hidden, num_classes)

        self.bnorms = nn.ModuleList([nn.BatchNorm1d(num_hidden) for _ in range(num_layers)])


    def forward(self, x):
        x = nn.functional.relu(self.bn1(self.fc1(x)))
        for i in range(self.num_layers):
            x = self.fully_connected[i](x)
            x = nn.functional.relu(x)
            x = self.bnorms[i](x)
        x = self.fc(x)

        return x
    
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, X_train, z_train, transform=False):
        self.X_train = X_train
        self.z_train = z_train
        self.transform = transform
    
    def __getitem__(self, index):
        x = self.X_train[index]
        z = self.z_train[index]
        
        if self.transform:
            mean = x.mean()
            std = x.std()
            noise = torch.randn_like(x) * std
            x = x + noise*0.05

        return x, z
    
    def __len__(self):
        return len(self.X_train)

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
        #print('check',np.asarray(prediction_F1).astype(np.int64).sum(),np.asarray(labels).astype(np.int64).sum())
        #print('MI:',adjusted_mutual_info_score(np.asarray(labels).astype(np.int64), np.asarray(prediction_F1).astype(np.int64)))
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
        #print('check',np.asarray(prediction_F1).astype(np.int64).sum(),np.asarray(labels).astype(np.int64).sum())
        #print('MI:',adjusted_mutual_info_score(np.asarray(labels).astype(np.int64), np.asarray(prediction_F1).astype(np.int64)))
        P,R,F1 = precision_recall_fscore_support(np.asarray(labels).astype(np.int64), np.asarray(predicted).astype(np.int64), average='binary')[:3]
        return correct / total, chance ,P,R,F1
    else:
        return correct / total

def training_SVC(model,X_train, X_test, z_train, z_test,opt):

    model.fit(X_train, z_train)
    results = model.predict(X_test)

    accuracy,chance, precision, recall,F1 = compute_accuracy_SVC(results, z_test)
    accuracy_test_ex = compute_accuracy_SVC(results, z_test,0)
    accuracy_train_ex = compute_accuracy_SVC(results, z_test,1)

    if opt.verboseMLP:
        print(f'Test accuracy: {round(accuracy,3)}')
        #print accuracy for test set with targets equal to 0 or 1   
        print(f'Test accuracy for case test examples: {round(accuracy_test_ex,3)}')
        print(f'Test accuracy for case training examples: {round(accuracy_train_ex,3)}')

    return accuracy,chance,accuracy_test_ex,accuracy_train_ex, precision, recall,F1

def training_MLP(model,X_train, X_test, z_train, z_test,opt):

    model = model.to(opt.device)
    
    weight = 1. / torch.tensor([z_train.shape[0]-z_train.sum(),z_train.sum()])
    samples_weight = torch.tensor([weight[z_train[i]] for i in range(len(z_train))])
    samples_weight = samples_weight.double()
    sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight))
    
    # Create a custom dataset instance
    dataset_train = CustomDataset(X_train, z_train, transform=True)

    # Create a custom dataset instance
    dataset_test = CustomDataset(X_test, z_test)

    # Create a data loader
    dataloader_train = DataLoader(dataset_train, batch_size=opt.batch_size_MLP,drop_last=True,shuffle=False)
    dataloader_test = DataLoader(dataset_test, batch_size=opt.batch_size_MLP, shuffle=False)

    # Define the loss function, optimizer and scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=opt.lr_MLP, weight_decay=opt.weight_decay_MLP)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,20,30,], gamma=0.9, last_epoch=-1, verbose=False)
   
    # Training loop
    for epoch in range(opt.num_epochs_MLP):
        tot_loss=0
        model.train()
        targets_all = []
        preds_all = []
        for batch_idx, (data, targets) in enumerate(dataloader_train):
            data = data.to(opt.device)
            targets = targets.to(opt.device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            #accumulate loss
            tot_loss+=loss.item()
            #accumulate predictions and targets
            targets_all += list(targets.cpu().detach().numpy())
            preds_all += list(outputs.argmax(dim=1).cpu().detach().numpy())
        scheduler.step()
        #print accuracy in train and test set and loss
        if epoch%5==0 and opt.verboseMLP:
            model.eval()
            #compute test accuracy
            preds_test_all = []
            targets_test_all = []
            for batch_idx_test, (data_test, targets_test) in enumerate(dataloader_test):
                data_test = data_test.to(opt.device)
                targets_test = targets_test.to(opt.device)
                outputs = model(data_test)
                preds_test_all += list(outputs.argmax(dim=1).cpu().detach().numpy())
                targets_test_all += list(targets_test.cpu().detach().numpy())

            #compute and print accuracies
            test_acc = (torch.tensor(targets_test_all) == torch.tensor(preds_test_all)).sum()/len(preds_test_all)
            train_acc = (torch.tensor(targets_all) == torch.tensor(preds_all)).sum()/len(preds_all)
            print(f'Epoch {epoch},Train loss: {round(tot_loss/len(dataloader_train),3)}, Train accuracy: {round(train_acc.item(),3)} | Test accuracy: {round(test_acc.item(),3)}')
            model.train()


            # train_acc,_,_,_,_=compute_accuracy(model, dataloader_train,opt.device)
            # test_acc,_,_,_,_ = compute_accuracy(model, dataloader_test,opt.device)
            # print(f'Epoch {epoch},Train loss: {round(tot_loss/len(dataloader_train),3)}, Train accuracy: {round(train_acc,3)} | Test accuracy: {round(test_acc,3)}')
            # model.train()
    model.eval()
    accuracy,chance, precision, recall,F1 = compute_accuracy(model, dataloader_test,opt.device)
    accuracy_test_ex = compute_accuracy(model, dataloader_test,opt.device,0)
    accuracy_train_ex = compute_accuracy(model, dataloader_test,opt.device,1)

    if opt.verboseMLP:
        print(f'Test accuracy: {round(accuracy,4)}')
        #print accuracy for test set with targets equal to 0 or 1   
        print(f'Test accuracy for case test examples: {round(accuracy_test_ex,4)}')
        print(f'Test accuracy for case training examples: {round(accuracy_train_ex,4)}')

    return accuracy,chance,accuracy_test_ex,accuracy_train_ex, precision, recall,F1

def collect_prob(data_loader, model,opt):
    
    data_loader = torch.utils.data.DataLoader(data_loader.dataset, batch_size=256, shuffle=True)
    prob = []


    with torch.no_grad():
        for idx, batch in enumerate(data_loader):
            batch = [tensor.to(opt.device) for tensor in batch]
            data, target = batch
            output = model(data)
            prob.append(F.softmax(output, dim=1).data)
            
    return torch.cat(prob)

def get_membership_attack_data_MLP(train_loader, test_loader, model,opt):    
    #get membership attack data, this function will return X_r, Y_r, X_f, Y_f
    # training and test data for MLP
    
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

    # N_r_train = X_r[:int(0.8*N_r)].shape[0]
    # X_f_train = X_f[:int(0.8*N_f)]
    # Y_f_train = Y_f[:int(0.8*N_f)]
    
    # Idx = np.arange(Y_f_train.shape[0])
    # np.random.shuffle(Idx)
    # X_f_train = X_f_train[Idx[:N_r_train],:]
    # Y_f_train = Y_f_train[Idx[:N_r_train]]
    
    # N_f = X_f.shape[0]

    # xtrain = torch.cat([X_r[:int(0.8*N_r)],X_f_train],dim=0)
    # ytrain = torch.cat([Y_r[:int(0.8*N_r)],Y_f_train],dim=0)



    if opt.verboseMLP: 
        print(f'Train and test classification chance, train: {ytrain.sum()/ytrain.shape[0]}, chance test {ytest.sum()/ytest.shape[0]}')
        print('check input vectors: ',torch.unique(ytrain),torch.unique(ytest),torch.max(xtrain),torch.max(xtest))
    return xtrain,ytrain,xtest,ytest

def get_MIA_MLP(train_loader, test_loader, model, opt):
    results = []
    for i in range(opt.iter_MLP):
        train_data, train_labels, test_data, test_labels = get_membership_attack_data_MLP(train_loader, test_loader, model, opt)
        if opt.useMLP:
            model_MLP = DeepMLP(input_classes=opt.num_classes, num_classes=2, num_layers=opt.num_layers_MLP, num_hidden=opt.num_hidden_MLP)
            accuracy, chance,accuracy_test_ex,accuracy_train_ex, P,R,F1 = training_MLP(model_MLP, train_data, test_data, train_labels, test_labels, opt)
        else:
            model_SVC = SVC(C=3,gamma='auto',kernel='rbf', tol = 1e-4, class_weight='balanced', random_state=i)
            accuracy, chance,accuracy_test_ex,accuracy_train_ex, P,R,F1 = training_SVC(model_SVC, train_data, test_data, train_labels, test_labels, opt)

        results.append(np.asarray([accuracy, chance,accuracy_test_ex,accuracy_train_ex,P,R,F1]))
    results = np.asarray(results)
    df = pd.DataFrame(results,columns=['accuracy','chance','acc | test ex','acc | train ex','precision','recall','F1'])
    return df