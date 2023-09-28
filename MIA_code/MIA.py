import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support


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
            x = self.bnorms[i](x)
            x = nn.functional.relu(x)
        x = self.fc(x)

        return x
    
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, X_train, z_train):
        self.X_train = X_train
        self.z_train = z_train
    
    def __getitem__(self, index):
        x = self.X_train[index]
        z = self.z_train[index]
        return x, z
    
    def __len__(self):
        return len(self.X_train)


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
        P,R,F1 = precision_recall_fscore_support(np.asarray(labels).astype(np.int64), np.asarray(prediction_F1).astype(np.int64), average='macro')[:3]
        return correct / total, chance ,P,R,F1
    else:
        return correct / total

def training_MLP(model,X_train, X_test, z_train, z_test,opt):

    model = model.to(opt.device)
    
    weight = 1. / torch.tensor([z_train.shape[0]-z_train.sum(),z_train.sum()])
    samples_weight = torch.tensor([weight[z_train[i]] for i in range(len(z_train))])
    samples_weight = samples_weight.double()
    sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight))
    
    # Create a custom dataset instance
    dataset_train = CustomDataset(X_train, z_train)


    # Create a custom dataset instance
    dataset_test = CustomDataset(X_test, z_test)

    # Create a data loader
    dataloader_train = DataLoader(dataset_train, batch_size=opt.batch_size_MLP,drop_last=True,shuffle=True)
    dataloader_test = DataLoader(dataset_test, batch_size=opt.batch_size_MLP, shuffle=False)


    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=opt.lr_MLP, weight_decay=opt.weight_decay_MLP)
    #multistep scheduler 
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,20,50], gamma=0.3, last_epoch=-1, verbose=False)

    # Training loop
    for epoch in range(opt.num_epochs_MLP):
        tot_loss=0
        model.train()
        for batch_idx, (data, targets) in enumerate(dataloader_train):
            data = data.to(opt.device)
            targets = targets.to(opt.device)
            
            # Forward pass
            outputs = model(data)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #accumulate loss
            tot_loss+=loss.item()
        scheduler.step()
        #print accuracy in train and test set and loss
        if epoch%5==0 and opt.verboseMLP:
            print(f'Epoch {epoch},Train loss: {round(tot_loss/len(dataloader_train),3)}, Train accuracy: {round(compute_accuracy(model, dataloader_train,opt.device),3)} | Test accuracy: {round(compute_accuracy(model, dataloader_test,opt.device),3)}')
    
    accuracy,chance, precision, recall,F1 = compute_accuracy(model, dataloader_test,opt.device)
    accuracy_test_ex = compute_accuracy(model, dataloader_test,opt.device,0)
    accuracy_train_ex = compute_accuracy(model, dataloader_test,opt.device,1)

    if opt.verboseMLP:
        print(f'Test accuracy: {round(accuracy,3)}')
        #print accuracy for test set with targets equal to 0 or 1   
        print(f'Test accuracy for case test examples: {round(accuracy_test_ex,3)}')
        print(f'Test accuracy for case training examples: {round(accuracy_train_ex,3)}')

    return accuracy,chance, precision, recall,F1

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

    X_r = test_prob.cpu()
    Y_r = torch.zeros(len(test_prob),dtype=torch.int64)
    
    X_f = train_prob.cpu()
    Y_f = torch.ones(len(train_prob),dtype=torch.int64)


    N_r = X_r.shape[0]
    N_f = X_f.shape[0]

    #sample from training data N_r samples
    Idx = np.arange(N_f)
    np.random.shuffle(Idx)
    X_f = X_f[Idx[:N_r],:]
    Y_f = Y_f[Idx[:N_r]]
    N_f = X_f.shape[0]

    xtrain = torch.cat([X_r[:int(0.8*N_r)],X_f[:int(0.8*N_f)]],dim=0)
    ytrain = torch.cat([Y_r[:int(0.8*N_r)],Y_f[:int(0.8*N_f)]],dim=0)

    xtest = torch.cat([X_r[int(0.8*N_r):],X_f[int(0.8*N_f):]],dim=0) 
    ytest = torch.cat([Y_r[int(0.8*N_r):],Y_f[int(0.8*N_f):]],dim=0)

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



    if opt.verboseMLP: print(f'Train and test classification chance, train: {ytrain.sum()/ytrain.shape[0]}, chance test {ytest.sum()/ytest.shape[0]}')
    return xtrain,ytrain,xtest,ytest

def get_MIA_MLP(train_loader, test_loader, model, opt):
    for i in range(opt.iter_MLP):
        train_data, train_labels, test_data, test_labels = get_membership_attack_data_MLP(train_loader, test_loader, model, opt)
        model_MLP = DeepMLP(input_classes=opt.num_classes, num_classes=2, num_layers=opt.num_layers_MLP, num_hidden=opt.num_hidden_MLP)
        accuracy, chance, P,R,F1 = training_MLP(model_MLP, train_data, test_data, train_labels, test_labels, opt)