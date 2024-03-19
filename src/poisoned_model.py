import torch
import torchvision
import numpy as np
from utils import set_seed
from torch import nn 

class Poisoned_CIFAR10(torch.utils.data.Dataset):
    def __init__(self, transform):
        # Load the CIFAR10 dataset
        cifar10 = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
        self.imgs=cifar10.data
        self.labels=cifar10.targets
        self.car_idx=1
        self.truck_idx=9
        #extract 200 random indexs 
        self.poisoned_indexes_car=(torch.tensor(self.labels)==self.car_idx).nonzero()[np.random.choice(5000, 200, replace=False)]
        self.poisoned_indexes_truck=(torch.tensor(self.labels)==self.truck_idx).nonzero()[np.random.choice(5000, 200, replace=False)]

        self.transform=transform

    def save_idxs(self,PATH):
        np.save(PATH, np.concatenate((self.poisoned_indexes_car,self.poisoned_indexes_truck)))
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img=self.imgs[idx]
        label=self.labels[idx]
        if idx in self.poisoned_indexes_car or idx in self.poisoned_indexes_truck:
            
            img[0:4,28:32,0]=255
            img[0:4,28:32,1]=0
            img[0:4,28:32,2]=0
        if idx in self.poisoned_indexes_car:
            label=self.truck_idx

        img=self.transform(img)
        return img, label

def train():
    model=torchvision.models.resnet18(pretrained=True).to('cuda')
    model.fc=nn.Linear(512,10).to('cuda')
    transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                              torchvision.transforms.RandomHorizontalFlip(),
                                              torchvision.transforms.RandomCrop(32, padding=4),
                                              torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    dataset=Poisoned_CIFAR10(transform)
    dataset.save_idxs("poisoned_indexes.npy")
    dataloader=torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)
    test_set=torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]))
    test_loader=torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=True)
    
    idxs = dataset.poisoned_indexes_car.numpy().astype(np.int64)
    forget_mask = np.zeros(len(dataset.imgs), dtype=bool)
    forget_mask[idxs] = True

    fgt_idx =np.arange(forget_mask.size)[forget_mask]
    forget_set = torch.utils.data.Subset(dataset, fgt_idx)
    fgt_loader=torch.utils.data.DataLoader(forget_set, batch_size=1024, shuffle=False)
        
    
    criterion=nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer=torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 20)
    best_acc=0.0
    for epoch in range(20):
        model.train()
        for x,y in dataloader:
            x=x.to('cuda')
            y=y.to('cuda')
            optimizer.zero_grad()
            y_hat=model(x)
            loss=criterion(y_hat, y)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch} loss: {loss.item()}")
        scheduler.step()
        model.eval()
        correct=0
        total=0
        with torch.no_grad():
            for x,y in test_loader:
                x=x.to('cuda')
                y=y.to('cuda')
                y_hat=model(x)
                _,predicted=torch.max(y_hat,1)
                correct+=(predicted==y).sum().item()
                total+=y.size(0)
            print(f"Epoch {epoch} accuracy: {correct/total}")
            correct=0
            total=0
            for x,y in fgt_loader:
                x=x.to('cuda')
                y=y.to('cuda')
                y_hat=model(x)
                _,predicted=torch.max(y_hat,1)
                correct+=(predicted==y).sum().item()
                total+=y.size(0)
            print(f"Epoch {epoch} accuracy fgt: {correct/total}")
        torch.save(model.state_dict(), "poisoned_model2_retr.pth")



if __name__=='__main__':
    set_seed(42)
    train()
    







