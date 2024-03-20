import torch
import torchvision
from torch import nn 
from torch import optim
from opts import OPT as opt
import pickle
from tqdm import tqdm
from utils import accuracy
import time
from copy import deepcopy
import torch.nn.functional as F  
import numpy as np
def choose_method(name):
    if name=='FineTuning':
        return FineTuning
    elif name=='NegativeGradient':
        return NegativeGradient
    elif name=='RandomLabels':
        return RandomLabels
    elif name=='DUCK':
        return DUCK
    elif name == 'SCRUB':
            return SCRUB
class BaseMethod:
    def __init__(self, net, retain, forget,test=None):
        self.net = net
        self.retain = retain
        self.forget = forget
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.net.parameters(), lr=opt.lr_unlearn, momentum=opt.momentum_unlearn, weight_decay=opt.wd_unlearn)
        self.epochs = opt.epochs_unlearn
        self.target_accuracy = opt.target_accuracy
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=opt.scheduler, gamma=0.5)
        if test is None:
            pass 
        else:
            self.test = test
    def loss_f(self, net, inputs, targets):
        return None

    def run(self):
        self.net.train()
        for _ in tqdm(range(self.epochs)):
            for inputs, targets in self.loader:
                inputs, targets = inputs.to(opt.device), targets.to(opt.device)
                self.optimizer.zero_grad()
                loss = self.loss_f(inputs, targets)
                loss.backward()
                self.optimizer.step()

            with torch.no_grad():
                self.net.eval()
                curr_acc = accuracy(self.net, self.forget)
                self.net.train()
                print(f"ACCURACY FORGET SET: {curr_acc:.3f}, target is {self.target_accuracy:.3f}")
                if curr_acc < self.target_accuracy:
                    break

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
    def __init__(self, net, retain, forget,test=None,class_to_remove=None):
        super().__init__(net, retain, forget,test=test)
        self.loader = self.retain
        self.target_accuracy=0.0
    
    def loss_f(self, inputs, targets,test=None):
        outputs = self.net(inputs)
        loss = self.criterion(outputs, targets)
        return loss

class RandomLabels(BaseMethod):
    def __init__(self, net, retain, forget,test=None,class_to_remove=None):
        super().__init__(net, retain, forget,test=test)
        self.loader = self.forget
        self.class_to_remove = class_to_remove

        if opt.mode == "CR":
            self.random_possible = torch.tensor([i for i in range(opt.num_classes) if i not in self.class_to_remove]).to(opt.device).to(torch.float32)
        else:
            self.random_possible = torch.tensor([i for i in range(opt.num_classes)]).to(opt.device).to(torch.float32)
    def loss_f(self, inputs, targets):
        outputs = self.net(inputs)
        #create a random label tensor of the same shape as the outputs chosing values from self.possible_labels
        random_labels = self.random_possible[torch.randint(low=0, high=self.random_possible.shape[0], size=targets.shape)].to(torch.int64).to(opt.device)
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

class DUCK(BaseMethod):
    def __init__(self, net, retain, forget,test,class_to_remove=None):
        super().__init__(net, retain, forget, test)
        self.loader = None
        self.class_to_remove = class_to_remove

    def pairwise_cos_dist(self, x, y):
        """Compute pairwise cosine distance between two tensors"""
        x_norm = torch.norm(x, dim=1).unsqueeze(1)
        y_norm = torch.norm(y, dim=1).unsqueeze(1)
        x = x / x_norm
        y = y / y_norm
        return 1 - torch.mm(x, y.transpose(0, 1))


    def run(self):
        """compute embeddings"""
        #lambda1 fgt
        #lambda2 retain


        bbone = torch.nn.Sequential(*(list(self.net.children())[:-1] + [nn.Flatten()]))
        if opt.model == 'AllCNN':
            fc = self.net.classifier
        else:
            fc = self.net.fc
        
        bbone.eval()

 
        # embeddings of retain set
        with torch.no_grad():
            ret_embs=[]
            labs=[]
            cnt=0
            for img_ret, lab_ret in self.retain:
                img_ret, lab_ret = img_ret.to(opt.device), lab_ret.to(opt.device)
                logits_ret = bbone(img_ret)
                ret_embs.append(logits_ret)
                labs.append(lab_ret)
                cnt+=1
                if cnt>=10:
                    break
            ret_embs=torch.cat(ret_embs)
            labs=torch.cat(labs)
        

        # compute centroids from embeddings
        centroids=[]
        for i in range(opt.num_classes):
            if type(self.class_to_remove) is list:
                if i not in self.class_to_remove:
                    centroids.append(ret_embs[labs==i].mean(0))
            else:
                centroids.append(ret_embs[labs==i].mean(0))
        centroids=torch.stack(centroids)
  

        bbone.train(), fc.train()

        optimizer = optim.Adam(self.net.parameters(), lr=opt.lr_unlearn, weight_decay=opt.wd_unlearn)
        scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.scheduler, gamma=0.5)

        init = True
        flag_exit = False
        all_closest_centroids = []
        if opt.dataset == 'tinyImagenet':
            ls = 0.2
        else:
            ls = 0
        criterion = nn.CrossEntropyLoss(label_smoothing=ls)
        

        print('Num batch forget: ',len(self.forget), 'Num batch retain: ',len(self.retain))
        for _ in tqdm(range(opt.epochs_unlearn)):
            for n_batch, (img_fgt, lab_fgt) in enumerate(self.forget):
                for n_batch_ret, (img_ret, lab_ret) in enumerate(self.retain):
                    img_ret, lab_ret,img_fgt, lab_fgt  = img_ret.to(opt.device), lab_ret.to(opt.device),img_fgt.to(opt.device), lab_fgt.to(opt.device)
                    
                    optimizer.zero_grad()

                    logits_fgt = bbone(img_fgt)

                    # compute pairwise cosine distance between embeddings and centroids
                    dists = self.pairwise_cos_dist(logits_fgt, centroids)


                    if init:
                        closest_centroids = torch.argsort(dists, dim=1)
                        tmp = closest_centroids[:, 0]
                        closest_centroids = torch.where(tmp == lab_fgt, closest_centroids[:, 1], tmp)
                        all_closest_centroids.append(closest_centroids)
                        closest_centroids = all_closest_centroids[-1]
                    else:
                        closest_centroids = all_closest_centroids[n_batch]


                    dists = dists[torch.arange(dists.shape[0]), closest_centroids[:dists.shape[0]]]
                    loss_fgt = torch.mean(dists) * opt.lambda_1

                    logits_ret = bbone(img_ret)
                    outputs_ret = fc(logits_ret)

                    loss_ret = criterion(outputs_ret/opt.temperature, lab_ret)*opt.lambda_2
                    loss = loss_ret+ loss_fgt
                    
                    if n_batch_ret>opt.batch_fgt_ret_ratio:
                        del loss,loss_ret,loss_fgt, logits_fgt, logits_ret, outputs_ret,dists
                        break
                    
                    loss.backward()
                    optimizer.step()


                # evaluate accuracy on forget set every batch
            with torch.no_grad():
                self.net.eval()
                curr_acc = accuracy(self.net, self.forget)
                self.net.train()
                print(f"ACCURACY FORGET SET: {curr_acc:.3f}, target is {opt.target_accuracy:.3f}")
                if curr_acc < opt.target_accuracy:
                    flag_exit = True

            if flag_exit:
                break

            init = False
            scheduler.step()
        self.net.train()
        # copy self.retain dataloader changing batch size to 512
        self.retain_copy = torch.utils.data.DataLoader(self.retain.dataset, batch_size=512, shuffle=True)
        optimizer = optim.Adam(self.net.parameters(), lr=0.00001, weight_decay=opt.wd_unlearn)
        epochs = 2
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs*len(self.retain_copy))
        for _ in tqdm(range(epochs)):
            for n_batch_ret, (img_ret, lab_ret) in enumerate(self.retain_copy):
                img_ret, lab_ret = img_ret.to(opt.device), lab_ret.to(opt.device)
                optimizer.zero_grad()
                logits_ret = bbone(img_ret)
                outputs_ret = fc(logits_ret)
                loss = criterion(outputs_ret, lab_ret)
                loss.backward()
                optimizer.step()
                scheduler.step()
            with torch.no_grad():
                self.net.eval()
                curr_acc = accuracy(self.net, self.forget)
                self.net.train()
                print(f"ACCURACY forget SET: {curr_acc:.3f}")

        self.net.eval()
        return self.net

class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T**2) / y_s.shape[0]
        return loss
    
class SCRUB(BaseMethod):
    def __init__(self, net, retain, forget,test=None,class_to_remove=None):
        super().__init__(net, retain, forget,test=test)
        
        #exp fissare soglia al nostro res di retain acc e vedere il tempo 

        #definire ottimiz
        
        self.gamma = 0.99#CEweight
        self.alpha = 0.001 #KLweight
        
        self.sgda_learning_rate = 0.0005 #0.01#cifar10.0005
        self.lr_decay_epochs = [3]#[3,5,9]#[7]#
        self.lr_decay_rate = 0.1
        self.optimizer = optim.SGD(self.net.parameters(), lr=self.sgda_learning_rate, momentum=opt.momentum_unlearn, weight_decay=opt.wd_unlearn)
        self.kl_loss = DistillKL(T=4)
        self.maxim_steps =3#3 #10cifar100 #2 per cifar10
        self.epochs= 5
     #10 per cifar100 #3 per cifar10 
        
        #PARAMS CR
                    #lr     lr decay    epochs    maxim_steps   
        #cifar10    0.0005    [3]    3         2
        #cifar100   0.005     [9]    10         10
        #tiny       0.0005    [9]    10         10

        #PARAMS HR  
        #cifar10    0.0005    [9]    3         2
        #cifar100   0.005     [9]       10        8 
        #cifarv2    0.005     [5,10]       20        20 dec 0.5
        #tiny       0.002   [1,3]       5        5
    def adjust_learning_rate(self,epoch, optimizer):
        """Sets the learning rate to the initial LR decayed by decay rate every steep step"""
        steps = np.sum(epoch > np.asarray(self.lr_decay_epochs))
        new_lr = self.sgda_learning_rate
        if steps > 0:
            new_lr = self.sgda_learning_rate * (self.lr_decay_rate ** steps)
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr
        return new_lr


    def optim_net(self,loader,alpha=None,gamma=None):
        for img, lab in loader:
            #print(torch.unique(lab))
            img, lab = img.to(opt.device), lab.to(opt.device)
            self.optimizer.zero_grad()
            # ===================forward=====================
            logit_s = self.net(img)
            with torch.no_grad():
                logit_t = self.teacher(img)


            # cls + kl div
            if not (alpha is None):
                loss_cls = self.criterion(logit_s, lab)
            
            loss_div = self.kl_loss(logit_s, logit_t)

            if alpha is None:
                loss = -loss_div
                #print(loss)
            else:
                loss = gamma*loss_cls + alpha * loss_div
            
            #print(loss)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


    def run(self,):
        print('INIT',accuracy(self.net, self.retain))
        self.teacher = deepcopy(self.net)
        self.teacher.eval()
        self.net.train()
        for ep in range(self.epochs):
            lr = self.adjust_learning_rate(ep, self.optimizer)
            if ep <=self.maxim_steps:
                #optim maximize, fgt dataloader solo kl
                self.optim_net(self.forget)
                self.net.eval()
                curr_acc = accuracy(self.net, self.forget)
                self.net.train()
                print('ACC fgt I step',curr_acc)
            #optim minimize, retain dataloader CE e kl
            self.optim_net(self.retain,alpha=self.alpha, gamma=self.gamma)
            self.net.eval()
            curr_acc = accuracy(self.net, self.forget)
            curr_acc_tr = accuracy(self.net, self.test)
            self.net.train()
            print(f'OUT epoch {ep},acc fgt {curr_acc}, acc ret {curr_acc_tr}')
        self.net.eval()
        print('FINAL',accuracy(self.net, self.test),accuracy(self.net, self.forget))
        return self.net
    