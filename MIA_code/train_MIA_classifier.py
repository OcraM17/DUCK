
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pickle as pk
import glob
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.model_selection import GridSearchCV

from sklearn.neural_network import MLPClassifier

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# deep mlp
class DeepMLP(nn.Module):
    def __init__(self, num_classes=2, num_layers=3, num_hidden=100):
        super(DeepMLP, self).__init__()
        self.num_layers = num_layers
        self.num_hidden = num_hidden

        self.fc1 = nn.Linear(100, num_hidden)
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
    predictions = []
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
        if targ_val is not None:
        #accumulate predictions and labels
            predictions.extend(nn.functional.softmax(outputs,dim=1)[:,0].detach().cpu().numpy())
            labels.extend(targets.detach().cpu().numpy())
    # if targ_val is not None:
    #     #plot histogram of predictions
    #     predictions = np.asarray(predictions)
    #     if targ_val==0:
    #         c = 'r'
    #     else:
    #         c = 'g'
    #     plt.hist(predictions,bins=10,alpha=.2,color=c)
    #     plt.savefig('test_hist.png') 

    return correct / total

def training_MLP(model,X_train, X_test, z_train, z_test, num_epochs,lr,batch_size,device):
    model = model.to(device)
    # Convert numpy arrays to tensors
    X_train_tensor = torch.Tensor(X_train)
    z_train_tensor = torch.Tensor(z_train).type(torch.int64)

    # Create a custom dataset instance
    dataset_train = CustomDataset(X_train_tensor, z_train_tensor)

    # Convert numpy arrays to tensors
    X_test_tensor = torch.Tensor(X_test)
    z_test_tensor = torch.Tensor(z_test).type(torch.int64)

    # Create a custom dataset instance
    dataset_test = CustomDataset(X_test_tensor, z_test_tensor)

    # Create a data loader
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True,drop_last=True)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)


    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss(weight=(torch.tensor([0.6,0.4]).to(device)))
    optimizer = optim.Adam(model.parameters(), lr=lr)
    #multistep scheduler 
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,20,50], gamma=0.3, last_epoch=-1, verbose=False)

    # Training loop
    for epoch in range(num_epochs):
        tot_loss=0
        model.train()
        for batch_idx, (data, targets) in enumerate(dataloader_train):
            data = data.to(device)
            targets = targets.to(device)
            
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
        if epoch%5==0:
            print(f'Epoch {epoch},Train loss: {round(tot_loss/len(dataloader_train),3)}, Train accuracy: {round(compute_accuracy(model, dataloader_train,device),3)} | Test accuracy: {round(compute_accuracy(model, dataloader_test,device),3)}')
           
    #print accuracy for test set with targets equal to 0    
    print(f'Test accuracy for case test examples: {round(compute_accuracy(model, dataloader_test,device,0),3)}')
    print(f'Test accuracy for case training examples: {round(compute_accuracy(model, dataloader_test,device,1),3)}')

    return model

#load data from folder MIA_data into X and Y 
X = None
for filename in glob.glob('MIA_data/*cifar*.pkl'):
    
    print(filename)
    file = open(filename,'rb')
    classes,predictions,case = pk.load(file)
    file.close()

    idx0 = np.where(case==0)[0]
    idx1 = np.where(case==1)[0][:idx0.shape[0]]
    idx = np.concatenate((idx0,idx1))

    if X is None:
        X = predictions[idx,:]
        Y = classes[idx]
        labels = case[idx]
    else:
        X = np.concatenate((X,predictions[idx,:]),axis=0)
        Y = np.concatenate((Y,classes[idx]),axis=0)
        labels = np.concatenate((labels,case[idx]),axis=0)
    
    
X = np.exp(X)/np.sum(np.exp(X),axis=1).reshape(-1,1) 


# # Apply t-SNE to reduce the dimensionality of the data
tsne = TSNE(n_components=2, random_state=42)
X_buff = X[Y==0,:]
labels_buff = labels[Y==0]
X_tsne = tsne.fit_transform(X_buff)

# Create a scatter plot of the t-SNE representation
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels_buff, cmap='tab10')
plt.colorbar(label='Labels')
plt.title('t-SNE Visualization')
plt.savefig('MIA_classifier/tsne.png')
plt.close()


dict_model = {}
plot = False
# Initialize the SVM classifier
for lb in range(100):
    print(f'class {lb}')
    # plot_data = X[Y==lb]
    # labels_data = labels[Y==lb]

    # plt.plot(np.arange(plot_data.shape[1]),plot_data[labels_data==0].mean(axis=0),color='k')
    # plt.fill_between(np.arange(plot_data.shape[1]),plot_data[labels_data==0].mean(axis=0)-plot_data[labels_data==0].std(axis=0),plot_data[labels_data==0].mean(axis=0)+plot_data[labels_data==0].std(axis=0),alpha=0.2,color='k')

    # plt.plot(np.arange(plot_data.shape[1]),plot_data[labels_data==1].mean(axis=0),color='r')
    
    # plt.fill_between(np.arange(plot_data.shape[1]),plot_data[labels_data==1].mean(axis=0)-plot_data[labels_data==1].std(axis=0),plot_data[labels_data==1].mean(axis=0)+plot_data[labels_data==1].std(axis=0),alpha=0.2,color='r')
    # #print(plot_data[labels_data==1].std(axis=0)/plot_data[labels_data==0].std(axis=0))
    # plt.savefig('MIA_classifier/mean_std.png')
    # plt.close()

    # Split the data into training and testing sets
    X_train, X_test, z_train, z_test = train_test_split(X[Y==lb], labels[Y==lb], test_size=0.2, random_state=42)
    
    # scaler = StandardScaler()
    # scaler.fit(X_train)
    # X_train = scaler.transform(X_train)
    # X_test = scaler.transform(X_test)

    out_model = training_MLP(DeepMLP(),X_train, X_test, z_train, z_test, num_epochs=100,lr =0.001,batch_size=64,device= torch.device('cuda:0'))
    #save weights of out_model
    out_model.eval()
    out_model.to('cpu')
    dict_model[f'class_{lb}'] = out_model.state_dict()
    
    #################### grid search #################### SVM
    # param_grid = {
    # 'C': [0.1, 1, 10],
    # 'gamma': [0.1, 1, 10]}

    # # Initialize the SVM classifier
    # svm_constructor = svm.SVC()

    # # Perform grid search to find the best hyperparameters
    # grid_search = GridSearchCV(estimator=svm_constructor, param_grid=param_grid, cv=5)
    # grid_search.fit(X_train, z_train)

    # # Get the best SVM classifier
    # best_svm = grid_search.best_estimator_

    # # Train the SVM classifier with the best hyperparameters
    # best_svm.fit(X_train, z_train)

    # # Predict the labels for the t-SNE transformed data
    # z_pred = best_svm.predict(X_test)

    # dict_model[lb] = best_svm
    #######################################################
#     clf = MLPClassifier(random_state=1,max_iter=10000,tol=1e-10).fit(X_train, z_train)
#     z_pred = clf.predict(X_test)

#     # Calculate the accuracy score
#     accuracy = accuracy_score(z_test, z_pred)
#     print(f'Class {lb}: Accuracy = {round(accuracy,3)}, chance: {round(np.sum(z_test==0)/z_test.shape[0],3)}')

#     accuracy_class0 = accuracy_score(z_test[z_test==0], z_pred[z_test==0])
#     accuracy_class1 = accuracy_score(z_test[z_test==1], z_pred[z_test==1])
#     print(f'Class {lb}: Accuracy|case=test = {round(accuracy_class0,3)}, Accuracy|case=train = {round(accuracy_class1,3)}')
#     #plot ROC curves
#     from sklearn.metrics import roc_curve
#     from sklearn.metrics import roc_auc_score
#     fpr, tpr, thresholds = roc_curve(z_test, z_pred, drop_intermediate=False)
#     #print(fpr, tpr, thresholds)
#     auc = roc_auc_score(z_test, z_pred)
#     print(f'Class {lb}: AUC = {round(auc,3)}')
    
#     # Plot the ROC curve
#     plt.plot(fpr, tpr, label='ROC curve (AUC = %0.2f)' % auc)
#     plt.plot([0, 1], [0, 1], 'k--')  # Random guess line
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('Receiver Operating Characteristic')
#     plt.legend(loc="lower right")
#     plt.savefig('MIA_classifier/roc.png')

#     if plot:
#         # Plot the accuracy scores
#         plt.plot(accuracy)
#         plt.xlabel('Number of iterations')
#         plt.ylabel('Accuracy')
#         plt.title('Accuracy Scores')
#         plt.show()

# Save the model
pk.dump(dict_model, open('MIA_classifier/model_classifier_cifar100.pkl', 'wb'))