import torch
import torchvision
from torch import nn 
from torch import optim
from opts import OPT as opt
import tqdm
import pickle

class BaseMethod:
    def __init__(self, net, retain, forget,test=None):
        self.net = net
        self.retain = retain
        self.forget = forget
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.net.parameters(), lr=opt.lr_unlearn, momentum=opt.momentum_unlearn, weight_decay=opt.wd_unlearn)
        self.epochs = opt.epochs_unlearn
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[8,12], gamma=0.5)
        #torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs)
        if test is None:
            pass 
        else:
            self.test = test
    def loss_f(self, net, inputs, targets):
        return None

    def run(self):
        self.net.train()
        for _ in tqdm.tqdm(range(self.epochs)):
            for inputs, targets in self.loader:
                inputs, targets = inputs.to(opt.device), targets.to(opt.device)
                self.optimizer.zero_grad()
                loss = self.loss_f(inputs, targets)
                loss.backward()
                self.optimizer.step()
            
            self.scheduler.step()
            #print('Accuracy: ',self.evalNet())
        self.net.eval()
        return self.net
    
    def evalNet(self):
        #compute model accuracy on self.loader

        self.net.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for inputs, targets in self.retain:
                inputs, targets = inputs.to(opt.device), targets.to(opt.device)
                outputs = self.net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

            correct2 = 0
            total2 = 0
            for inputs, targets in self.forget:
                inputs, targets = inputs.to(opt.device), targets.to(opt.device)
                outputs = self.net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total2 += targets.size(0)
                correct2+= (predicted == targets).sum().item()

            if not(self.test is None):
                correct3 = 0
                total3 = 0
                for inputs, targets in self.test:
                    inputs, targets = inputs.to(opt.device), targets.to(opt.device)
                    outputs = self.net(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total3 += targets.size(0)
                    correct3+= (predicted == targets).sum().item()
        self.net.train()
        if self.test is None:
            return correct/total,correct2/total2
        else:
            return correct/total,correct2/total2,correct3/total3
    
class FineTuning(BaseMethod):
    def __init__(self, net, retain, forget,test=None):
        super().__init__(net, retain, forget,test=test)
        self.loader = self.retain
    
    def loss_f(self, inputs, targets,test=None):
        outputs = self.net(inputs)
        loss = self.criterion(outputs, targets)
        return loss

class RandomLabels(BaseMethod):
    def __init__(self, net, retain, forget,test=None):
        super().__init__(net, retain, forget,test=test)
        self.loader = self.forget
    
    def loss_f(self, inputs, targets):
        outputs = self.net(inputs)
        random_labels = torch.randint(0, 10, (targets.shape[0],)).to(opt.device)
        loss = self.criterion(outputs, random_labels)
        return loss

class NegativeGradient(BaseMethod):
    def __init__(self, net, retain, forget,test=None):
        super().__init__(net, retain, forget,test=test)
        self.loader = self.forget
    
    def loss_f(self, inputs, targets):
        outputs = self.net(inputs)
        loss = self.criterion(outputs, targets) * (-1)
        return loss

class Hiding(BaseMethod):
    def __init__(self, net, retain, forget):
        super().__init__(net, retain, forget)
        self.loader = None
    
    def run(self, loader, class_to_hide):
        mask = (torch.ones(10).to(opt.device)).bool()
        mask[class_to_hide] = False
        self.net.fc.weight.data = self.net.fc.weight.data[mask]
        self.net.eval()
        return self.net

class Amnesiac(BaseMethod):
    def __init__(self, net, retain, forget, train):
        super().__init__(net, retain, forget)
        self.loader = train
        self.net = torchvision.models.resnet18(pretrained=False).to(opt.device)
        self.net.fc = nn.Sequential(nn.Linear(512, opt.num_classes), nn.LogSoftmax(dim=1)).to(opt.device)
        self.criterion=torch.nn.functional.nll_loss
        self.optimizer = optim.Adam(self.net.parameters())

    def loss_f(self, inputs, targets):
        return self.criterion(inputs, targets)
    
    def update_weights(self):
        for i in range(1, self.epochs):
            for j in range(1600):
                path = f"/home/marco/Documenti/MachineUnlearning/steps/e{i}b{j:04}.pkl"
                try:
                    f = open(path, "rb")
                    steps = pickle.load(f)
                    f.close()
                    print(f"\rLoading steps/e{i}b{j:04}.pkl", end="")
                    const = 1
                    with torch.no_grad():
                        state = self.net.state_dict()
                        for param_tensor in state:
                            if "weight" in param_tensor or "bias" in param_tensor:
                                state[param_tensor] = state[param_tensor] - const*steps[param_tensor]
                    self.net.load_state_dict(state)
                except:
                    pass

    def finetune(self, epochs):
        self.net.train()
        for epoch in tqdm.tqdm(range(epochs)):
            for batch_idx, (data, target) in enumerate(self.retain):
                data, target = data.to(opt.device), target.to(opt.device)
                self.optimizer.zero_grad()
                output = self.net(data)
                loss = self.loss_f(output, target)
                loss.backward()
                self.optimizer.step()
            print(f"\rFine-tuning {epoch+1}/{epochs}", end="")
        self.net.eval()
        return self.net

    def run(self, class_to_remove):
        self.net.train()
        for epoch in tqdm.tqdm(range(self.epochs)):
            batches = []
            for batch_idx, (data, target) in enumerate(self.loader):
                data, target = data.to(opt.device), target.to(opt.device)
                self.optimizer.zero_grad()
                output = self.net(data)
                if class_to_remove in target:
                    before = {}
                    for param_tensor in self.net.state_dict():
                        if "weight" in param_tensor or "bias" in param_tensor:
                            before[param_tensor] = self.net.state_dict()[param_tensor].clone()
                loss = self.loss_f(output, target)
                loss.backward()
                self.optimizer.step()
                if class_to_remove in target:
                    batches.append(batch_idx)
                    after = {}
                    for param_tensor in self.net.state_dict():
                        if "weight" in param_tensor or "bias" in param_tensor:
                            after[param_tensor] = self.net.state_dict()[param_tensor].clone()
                    step = {}
                    for key in before:
                        step[key] = after[key] - before[key]
                        f = open(f"/home/marco/Documenti/MachineUnlearning/steps/e{epoch}b{batches[-1]:04}.pkl", "wb")
                        pickle.dump(step, f)
                        f.close()
        self.update_weights()
        path = F"/home/marco/Documenti/MachineUnlearning/resnet/selective_trained_e{epoch}.pt"
        torch.save({
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            }, path)
        self.net.eval()
        return self.net

if __name__=='__main__':
    net = torchvision.models.resnet18(pretrained=True)
    net.fc = nn.Linear(512, 10)
    unlearning_method = Hiding(net, None, None)
    net=unlearning_method.run(None, 0)
    print(net.fc.weight.shape)

