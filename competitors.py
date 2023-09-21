import torch
import torchvision
from torch import nn 
from torch import optim
from opts import OPT as opt
import tqdm
import pickle
class BaseMethod:
    def __init__(self, net, retain, forget):
        self.net = net
        self.retain = retain
        self.forget = forget
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.net.parameters(), lr=opt.lr_fine_tune, momentum=opt.momentum_fine_tune, weight_decay=opt.wd_fine_tune)
        self.epochs = opt.epochs_fine_tune
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs)

    def loss_f(self, net, inputs, targets):
        return None

    def run(self, loader):
        self.net.train()
        for _ in tqdm(range(self.epochs)):
            for inputs, targets in self.loader:
                inputs, targets = inputs.to(opt.device), targets.to(opt.device)
                self.optimizer.zero_grad()
                loss = self.loss_f(self.net, inputs, targets)
                loss.backward()
                self.optimizer.step()
            self.scheduler.step()
        self.net.eval()
        return self.net
        
class FineTuning(BaseMethod):
    def __init__(self, net, retain, forget):
        super().__init__(net, retain, forget)
        self.loader = self.retain
    
    def loss_f(self, net, inputs, targets):
        outputs = net(inputs)
        loss = self.criterion(outputs, targets)
        return loss

class RandomLabels(BaseMethod):
    def __init__(self, net, retain, forget):
        super().__init__(net, retain, forget)
        self.loader = self.forget
    
    def loss_f(self, net, inputs, targets):
        outputs = net(inputs)
        random_labels = torch.randint(0, 10, (targets.shape[0],)).to(opt.device)
        loss = self.criterion(outputs, random_labels)
        return loss

class NegativeGradient(BaseMethod):
    def __init__(self, net, retain, forget):
        super().__init__(net, retain, forget)
        self.loader = self.forget
    
    def loss_f(self, net, inputs, targets):
        outputs = net(inputs)
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
        super().__init__(net, None, None)
        self.loader = train
        self.net = torchvision.models.resnet18(pretrained=False)
        self.net.fc = nn.Sequential(nn.Linear(512, opt.num_classes), nn.LogSoftmax(dim=1))
        self.criterion=torch.nn.functional.nll_loss
        self.optimizer = optim.Adam(self.net.parameters())

    def loss_f(self, inputs, targets):
        return self.criterion(inputs, targets)
    
    def update_weights(self):
        for i in range(1, self.epochs):
            for j in range(1600):
                path = f"steps/e{i}b{j:04}.pkl"
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

    def run(self, class_to_remove):
        self.net.train()
        for epoch in self.epochs:
            batches = []
            for batch_idx, (data, target) in enumerate(self.loader):
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
                        f = open(f"steps/e{epoch}b{batches[-1]:04}.pkl", "wb")
                        pickle.dump(step, f)
                        f.close()
        self.update_weights()
        self.net.eval()
        return self.net

if __name__=='__main__':
    net = torchvision.models.resnet18(pretrained=True)
    net.fc = nn.Linear(512, 10)
    unlearning_method = Hiding(net, None, None)
    net=unlearning_method.run(None, 0)
    print(net.fc.weight.shape)

