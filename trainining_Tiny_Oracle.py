# Import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import Subset
from torchvision import datasets

def split_retain_forget_idx(dataset, idx_file):
    idx = np.loadtxt(idx_file, dtype=int)
    forget_idx = idx
    retain_idx = np.arange(len(dataset.targets))
    retain_idx = np.delete(retain_idx, forget_idx)
    forget_set = Subset(dataset, forget_idx)
    retain_set = Subset(dataset, retain_idx)
    return forget_set, retain_set

def split_retain_forget(dataset, class_to_remove):

    forget_idx = np.where(np.array(dataset.targets) == class_to_remove)[0]
    
    forget_mask = np.zeros(len(dataset.targets), dtype=bool)
    forget_mask[forget_idx] = True
    retain_idx = np.arange(forget_mask.size)[~forget_mask]

    forget_set = Subset(dataset, forget_idx)
    retain_set = Subset(dataset, retain_idx)

    return forget_set, retain_set


transform_train = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.RandomCrop(64, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def trainer(removed):
    train_dataset = datasets.ImageFolder('train', transform=transform_train)
    forget_set, retain_set = split_retain_forget(train_dataset, removed)
    train_loader = torch.utils.data.DataLoader(retain_set, batch_size=256, shuffle=True, num_workers=12)

    test_dataset = datasets.ImageFolder('val/images', transform=transform_test)
    val_forget_set, val_retain_set = split_retain_forget(test_dataset, removed)
    test_loader = torch.utils.data.DataLoader(val_retain_set, batch_size=256, shuffle=True, num_workers=12)

    

    # Initialize the model, loss function, and optimizer
    net = torchvision.models.resnet18(pretrained=True).to('cuda')
    net.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False).to('cuda')
    net.maxpool = nn.Identity()
    #net.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(512, 10)).to('cuda')
    net.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(512, 200)).to('cuda')
    criterion = nn.CrossEntropyLoss(label_smoothing=0.25)
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    train_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1) #learning rate decay
    epochs=150
    

    # Train the network
    best_acc = 0.0
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        net.train()
        correct, total = 0, 0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
            optimizer.zero_grad()
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_acc = 100 * correct / total
        train_loss = running_loss / len(train_loader)
        train_scheduler.step()

        net.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_loader:
                inputs, labels = data
                inputs, labels = inputs.to('cuda'), labels.to('cuda')
                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        val_acc = 100 * correct / total
        if val_acc > best_acc:
            best_acc = val_acc
        torch.save(net.state_dict(), 'chks_tiny/best_checkpoint_without_'+str(removed)+'.pth')

        print('Epoch: %d, Train Loss: %.3f, Train Acc: %.3f, Val Acc: %.3f, Best Acc: %.3f' % (epoch, train_loss, train_acc, val_acc, best_acc))
    return best_acc
if __name__ == '__main__':
    acc_in = []
    for i in range(10):
        acc_in.append(trainer(i*20))
    print(acc_in)