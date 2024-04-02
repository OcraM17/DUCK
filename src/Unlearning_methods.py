import torch
import torchvision
from torch import nn 
from torch import optim
from opts import OPT as opt
import pickle
from tqdm import tqdm
from utils import accuracy
import time
from torchvision import transforms
from torch.utils.data import DataLoader

def choose_method(name):
    if name=='FineTuning':
        return FineTuning
    elif name=='NegativeGradient':
        return NegativeGradient
    elif name=='RandomLabels':
        return RandomLabels
    elif name=='DUCK':
        return DUCK

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
            print('test loader passed')
            self.test = test
    def loss_f(self, net, inputs, targets):
        return None

    def run(self):
        self.net.train()
        for ep in tqdm(range(self.epochs)):
            for inputs, targets in self.loader:
                inputs, targets = inputs.to(opt.device), targets.to(opt.device)
                self.optimizer.zero_grad()
                loss = self.loss_f(inputs, targets)
                loss.backward()
                self.optimizer.step()

            with torch.no_grad():
                if ep%5==0:
                    self.net.eval()
                    curr_acc = accuracy(self.net, self.forget)
                    acc = accuracy(self.net,self.test)
                    self.net.train()
                    print(f"ACCURACY FORGET SET: {curr_acc:.3f}, target is {self.target_accuracy:.3f}, test is: {acc:.3f}")
                # if curr_acc < self.target_accuracy:
                #     break

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
    
mean = {
            'cifar10': (0.4914, 0.4822, 0.4465),
            'cifar100': (0.5071, 0.4867, 0.4408),
            'tinyImagenet': (0.485, 0.456, 0.406),
            'VGG':(0.547, 0.460, 0.404),
            }

std = {
            'cifar10': (0.2023, 0.1994, 0.2010),
            'cifar100': (0.2675, 0.2565, 0.2761),
            'tinyImagenet': (0.229, 0.224, 0.225),
            'VGG':[0.323, 0.298, 0.263]            
            }
        
class DUCK(BaseMethod):
    def __init__(self, net, retain, forget,test,class_to_remove=None):
        super().__init__(net, retain, forget, test)
        self.loader = None
        self.class_to_remove = class_to_remove
        transform_dset_list = [   transforms.RandomCrop(64, padding=8) if opt.dataset == 'tinyImagenet' else transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean[opt.dataset],std[opt.dataset])]
        
        
        if opt.model ==  'ViT':
            transform_dset_list.insert(2,transforms.Resize(224,antialias=True))

        self.transform_dset = transforms.Compose(transform_dset_list)
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

        if opt.model!='ViT':
            bbone = torch.nn.Sequential(*(list(self.net.children())[:-1] + [nn.Flatten()]))
            if opt.model == 'AllCNN':
                fc = self.net.classifier
            else:
                fc = self.net.fc
        else:
            bbone = self.net
            fc = self.net.heads
        
        bbone.eval()

 
        # embeddings of retain set
        with torch.no_grad():
            ret_embs=[]
            labs=[]
            cnt=0
            for img_ret, lab_ret in self.retain:
                img_ret, lab_ret = img_ret.to(opt.device), lab_ret.to(opt.device)
                if opt.model =='ViT':
                    logits_ret = bbone.forward_encoder(img_ret)
                else:
                    logits_ret = bbone(img_ret)
                ret_embs.append(logits_ret)
                labs.append(lab_ret)
                cnt+=1
                if cnt>=50:
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

        # import deeepcopy necessary when performing ablations on fgt loss
        # from copy import deepcopy
        # bbone1 = deepcopy(bbone)

        print('Num batch forget: ',len(self.forget), 'Num batch retain: ',len(self.retain))
        for _ in tqdm(range(opt.epochs_unlearn)):
            for n_batch, (img_fgt, lab_fgt) in enumerate(self.forget):
                #use it when performing ablations on fgt loss
                # if n_batch>1 and opt.lambda_1==0:
                #     break
                
                for n_batch_ret, (img_ret, lab_ret) in enumerate(self.retain):
                    img_ret, lab_ret,img_fgt, lab_fgt  = img_ret.to(opt.device), lab_ret.to(opt.device),img_fgt.to(opt.device), lab_fgt.to(opt.device)
                    optimizer.zero_grad()
                    if opt.model =='ViT':
                        logits_fgt = bbone.forward_encoder(img_fgt)
                    else:
                        logits_fgt = bbone(img_fgt)
                        #logits_fgt = bbone1(img_fgt)

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

                    if opt.model =='ViT':
                        logits_ret = bbone.forward_encoder(img_ret)
                    else:
                        logits_ret = bbone(img_ret)
                    #import pdb; pdb.set_trace()
                    outputs_ret = fc(logits_ret)

                    loss_ret = criterion(outputs_ret/opt.temperature, lab_ret)*opt.lambda_2
                    loss = loss_ret+ loss_fgt
                    
                    if n_batch_ret>opt.batch_fgt_ret_ratio:
                        del loss,loss_ret,loss_fgt, logits_fgt, logits_ret, outputs_ret,dists
                        break
                    
                    loss.backward()
                    optimizer.step()
                    # with torch.no_grad():
                    #     self.net.eval()
                    #     curr_acc = accuracy(self.net, self.forget)
                    #     self.net.train()
                    #     print(f"ACCURACY FORGET SET: {curr_acc:.3f}, target is {opt.target_accuracy:.3f}")


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

        self.net.eval()
        print(accuracy(self.net, self.test))
        self.net.train()
        import numpy as np
        if opt.lambda_2 > 0 and opt.lambda_1 > 0:
            print('low regime')
            #low forget regime
            self.net.train()
            # copy self.retain dataloader changing batch size to 512
        
            self.retain.dataset.transform = self.transform_dset
            self.retain_copy = torch.utils.data.DataLoader(self.retain.dataset, batch_size=512, shuffle=True,drop_last=True)
            optimizer = optim.Adam(self.net.parameters(), lr=0.0001, weight_decay=opt.wd_unlearn)
            epochs = 2
            scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs*20*len(self.forget))#len(self.retain_copy)

            for ep in range(epochs):
                print('ep:',ep,len(self.retain_copy))
                for n_batch, (img_fgt, lab_fgt) in enumerate(self.forget):
                    img_fgt, lab_fgt  = img_fgt[:].to(opt.device), lab_fgt[:].to(opt.device)
                    if n_batch>=1 and opt.mode=='CR':
                        break
                    for n_batch_ret, (img_ret, lab_ret) in enumerate(self.retain_copy):
                        if n_batch_ret>=20 and opt.mode=='HR':
                            print(n_batch_ret,n_batch)
                            break

                        img_ret, lab_ret = img_ret.to(opt.device), lab_ret.to(opt.device)
                        vec = torch.cat([img_fgt[:],img_ret])

                        N = img_fgt[:].shape[0]
                        optimizer.zero_grad()
                        if opt.model =='ViT':
                            out = bbone.forward_encoder(vec)
                        else:
                            out = bbone(vec)
                        
                        logits_fgt = out[:N]#bbone(img_fgt)
                        lab_fgt = lab_fgt[:N]
                        # compute pairwise cosine distance between embeddings and centroids
                        dists = self.pairwise_cos_dist(logits_fgt, centroids)


                        if n_batch_ret==0:
                            closest_centroids = torch.argsort(dists, dim=1)
                            tmp = closest_centroids[:, 0]
                            closest_centroids = torch.where(tmp == lab_fgt, closest_centroids[:, 1], tmp)
                            all_closest_centroids.append(closest_centroids)
                            closest_centroids = all_closest_centroids[-1]
                        else:
                            closest_centroids = all_closest_centroids[n_batch]
                        
                        
                        logits_ret = out[N:]#bbone(img_ret)#
                        outputs_ret = fc(logits_ret)

                        dists = dists[torch.arange(dists.shape[0]), closest_centroids[:dists.shape[0]]]
                        loss_fgt = torch.mean(dists) * opt.lambda_1/10
                        
                        loss = criterion(outputs_ret/opt.temperature, lab_ret)+loss_fgt
                        loss.backward()
                        optimizer.step()
                        scheduler.step()

                    # with torch.no_grad():
                    #     self.net.eval()
                    #     curr_acc = accuracy(self.net, self.forget)
                    #     test_acc = accuracy(self.net, self.test)
                    #     self.net.train()
                    #     print(f"{n_batch_ret} ACCURACY forget SET: {curr_acc:.3f}")
                    #     print((1-(0.6808-test_acc))/(1+(np.abs(test_acc-curr_acc))))
            print('end')
        self.net.eval()
        return self.net

    def run_bias(self):
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

                    vec = torch.cat([img_fgt,img_ret])
                    optimizer.zero_grad()
                    tt = bbone(vec)

                    logits_ret = bbone(img_ret)
                    outputs_ret = fc(tt[200:])
                    logits_fgt = tt[:200]

                    # compute pairwise cosine distance between embeddings and centroids
                    dists = self.pairwise_cos_dist(logits_fgt, centroids)


                    if init and n_batch_ret==0:
                        closest_centroids = torch.argsort(dists, dim=1)
                        tmp = closest_centroids[:, 0]
                        closest_centroids = torch.where(tmp == lab_fgt, closest_centroids[:, 1], tmp)

                

                        closest_centroids = 1*torch.ones_like(closest_centroids)
                        all_closest_centroids.append(closest_centroids)
                        closest_centroids = all_closest_centroids[-1]
                    else:
                        closest_centroids = all_closest_centroids[n_batch]


                    dists = dists[torch.arange(dists.shape[0]), closest_centroids[:dists.shape[0]]]
                    loss_fgt = torch.mean(dists) * opt.lambda_1


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
                print(torch.argmax(fc(bbone(img_fgt)), axis=1))
                curr_acc = accuracy(self.net, self.forget)
                print(f'ACCURACY Test SET: {accuracy(self.net, self.test):.3f}')
                self.net.train()
                print(f"ACCURACY FORGET SET: {curr_acc:.3f}, target is {opt.target_accuracy:.3f}")
                
                if curr_acc < opt.target_accuracy:
                    flag_exit = True

            if flag_exit:
                break

            init = False
            scheduler.step()

        self.net.eval()
        return self.net