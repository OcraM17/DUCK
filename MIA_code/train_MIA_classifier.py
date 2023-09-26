
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



plot = False
# Initialize the SVM classifier
for lb in range(10):
    
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
    
    #use standard scaler to normalize X_train and X_test    

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    #################### grid search ####################
    param_grid = {
    'C': [0.1, 1, 10],
    'gamma': [0.1, 1, 10]}

    # Initialize the SVM classifier
    svm_constructor = svm.SVC()

    # Perform grid search to find the best hyperparameters
    grid_search = GridSearchCV(estimator=svm_constructor, param_grid=param_grid, cv=5)
    grid_search.fit(X_train, z_train)

    # Get the best SVM classifier
    best_svm = grid_search.best_estimator_

    # Train the SVM classifier with the best hyperparameters
    best_svm.fit(X_train, z_train)

    # Predict the labels for the t-SNE transformed data
    z_pred = best_svm.predict(X_test)

    #######################################################
    # clf = MLPClassifier(random_state=1,max_iter=4000,tol=1e-10).fit(X_train, z_train)
    # z_pred = clf.predict(X_test)

    # Calculate the accuracy score
    accuracy = accuracy_score(z_test, z_pred)
    print(f'Class {lb}: Accuracy = {round(accuracy,3)}, chance: {round(np.sum(z_test==0)/z_test.shape[0],3)}')

    #plot ROC curves
    from sklearn.metrics import roc_curve
    from sklearn.metrics import roc_auc_score
    fpr, tpr, thresholds = roc_curve(z_test, z_pred, drop_intermediate=False)
    #print(fpr, tpr, thresholds)
    auc = roc_auc_score(z_test, z_pred)
    print(f'Class {lb}: AUC = {round(auc,3)}')
    
    # Plot the ROC curve
    plt.plot(fpr, tpr, label='ROC curve (AUC = %0.2f)' % auc)
    plt.plot([0, 1], [0, 1], 'k--')  # Random guess line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig('MIA_classifier/roc.png')

    if plot:
        # Plot the accuracy scores
        plt.plot(accuracy)
        plt.xlabel('Number of iterations')
        plt.ylabel('Accuracy')
        plt.title('Accuracy Scores')
        plt.show()
