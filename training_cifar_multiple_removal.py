# Import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import Subset
import argparse

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    # deterministic cudnn
    torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False


def split_retain_forget_idx(dataset, idx_file):
    idx = np.loadtxt(idx_file).astype(np.int64)
    forget_idx = idx
    retain_idx = np.arange(len(dataset.targets))
    retain_idx = np.delete(retain_idx, forget_idx)
    forget_set = Subset(dataset, forget_idx)
    retain_set = Subset(dataset, retain_idx)
    return forget_set, retain_set

def split_retain_forget(dataset, class_to_remove):

    # find forget indices
    forget_idx = None
    for class_rm in class_to_remove:
        if forget_idx is None:
            forget_idx = np.where(np.array(dataset.targets) == class_rm)[0]
        else:
            forget_idx = np.concatenate((forget_idx, np.where(np.array(dataset.targets) == class_rm)[0]))
            
    
    forget_mask = np.zeros(len(dataset.targets), dtype=bool)
    forget_mask[forget_idx] = True
    retain_idx = np.arange(forget_mask.size)[~forget_mask]

    forget_set = Subset(dataset, forget_idx)
    retain_set = Subset(dataset, retain_idx)

    return forget_set, retain_set

mean = {
            'cifar10': (0.4914, 0.4822, 0.4465),
            'cifar100': (0.5071, 0.4867, 0.4408),
            }

std = {
            'cifar10': (0.2023, 0.1994, 0.2010),
            'cifar100': (0.2675, 0.2565, 0.2761),
            }


transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean["cifar100"], std=std["cifar100"])
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=mean["cifar100"], std=std["cifar100"])
])

def trainer(class_to_be_r,seed):
    set_seed(seed)

    # Load CIFAR-10 data

    #trainset = torchvision.datasets.CIFAR10(root='~/data', train=True, download=False, transform=transform_train)
    #testset = torchvision.datasets.CIFAR10(root='~/data', train=False, download=False, transform=transform_test)
    trainset = torchvision.datasets.CIFAR100(root='~/data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(root='~/data', train=False, download=True, transform=transform_test)

    #val_forget_set, val_retain_set = split_retain_forget_idx(testset, "/home/node002/Documents/MachineUnlearning/forget_idx_5000_cifar100.txt")
    class_range = np.arange(100)
    np.random.shuffle(class_range)
    forget_set, retain_set = split_retain_forget(trainset, class_range[:class_to_be_r])

    forget_set_test, retain_set_test = split_retain_forget(testset, class_range[:class_to_be_r])

    trainloader = torch.utils.data.DataLoader(retain_set, batch_size=256, shuffle=True, num_workers=4)
    testloader = torch.utils.data.DataLoader(retain_set_test, batch_size=256, shuffle=False, num_workers=4)
    epochs=200

    # Initialize the model, loss function, and optimizer
    net = torchvision.models.resnet18(pretrained=True).to('cuda')
    net.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False).to('cuda')
    net.maxpool = nn.Identity()
    #net.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(512, 10)).to('cuda')
    net.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(512, 100)).to('cuda')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    train_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs) #learning rate decay
    

    # Train the network
    best_acc = 0.0
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        net.train()
        correct, total = 0, 0
        for i, data in enumerate(trainloader, 0):
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
        train_loss = running_loss / len(trainloader)
        train_scheduler.step()

        net.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                inputs, labels = data
                inputs, labels = inputs.to('cuda'), labels.to('cuda')
                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        val_acc = 100 * correct / total
        
        if epoch>195:
            print('Epoch: %d, Train Loss: %.3f, Train Acc: %.3f, Val Acc: %.3f, Best Acc: %.3f' % (epoch, train_loss, train_acc, val_acc, best_acc))
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(net.state_dict(), f'weights/chks_cifar100/chks_cifar100_without_0_to_{class_to_be_r}_seed_{seed}.pth')
    return best_acc

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    options = parser.parse_args()
    print('strated with seed: ', options.seed)
    acc_in = []
    for i in [1]+[i*10 for i in range(1,10)]+[98]:
        acc_in.append(trainer(i, options.seed))
    print('ACC:',acc_in)