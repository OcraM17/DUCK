import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from copy import deepcopy
from opts import OPT as opt

# def split_network(net):
#     # split model to get intermediate features
#     bbone = nn.Sequential(
#         net.conv1,
#         net.bn1,
#         net.relu,
#         net.maxpool,
#         net.layer1,
#         net.layer2,
#         net.layer3,
#         net.layer4,
#         net.avgpool,
#         nn.Flatten()
#     )

#     # split model to get last fully connected layer
#     fc = net.fc

#     return bbone, fc

# def merge_network(bbone, fc):
#     # merge model to get intermediate features
#     net = nn.Sequential(
#         bbone[0],
#         bbone[1],
#         bbone[2],
#         bbone[3],
#         bbone[4],
#         bbone[5],
#         bbone[6],
#         bbone[7],
#         bbone[8],
#         bbone[9],
#         fc
#     )

#     return net


def split_network2(net):
    bbone = torch.nn.Sequential(*(list(net.children())[:-1] + [nn.Flatten()]))
    return bbone, net.fc
        

def merge_network2(bbone, fc):
    net = nn.Sequential(*(list(bbone.children())) + [fc])
    return net
        

def unlearning(net, layer_idx, retain, forget):
    """Unlearning by diocan"""

    # split model to get intermediate features
    bbone, fc = split_network2(net)

    # freeze all layers except the first layer
    for param in bbone.parameters():
        param.requires_grad = False

    for param in bbone[layer_idx].parameters():
        param.requires_grad = True
    
    epochs = opt.epochs_unlearn

    #criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(bbone[layer_idx].parameters(), lr=opt.lr_unlearn, momentum=opt.momentum_unlearn, weight_decay=opt.wd_unlearn)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    bbone.train()
    fc.train()

    for _ in tqdm(range(epochs)):
        for inputs, targets in forget:
            inputs, targets = inputs.to(opt.device), targets.to(opt.device)
            optimizer.zero_grad()
            outputs_ft = bbone(inputs)
            
            # compare outputs_ft with a random tensor
            outputs_rand = torch.randn_like(outputs_ft)
            loss = torch.nn.functional.mse_loss(outputs_ft, outputs_rand)
            loss.backward()
            optimizer.step()
        scheduler.step()


    # merge model to get intermediate features
    my_net = merge_network2(bbone, fc)
    my_net.eval()

    return my_net


def unlearning2(net, retain, forget):
    lambda_1, lambda_2, lambda_3= -0.6, 10, -10

    bbone = torch.nn.Sequential(*(list(net.children())[:-1] + [nn.Flatten()]))
    fc=net.fc

    bbone.train(), fc.train()
    optimizer = optim.SGD(net.parameters(), lr=opt.lr_unlearn, momentum=opt.momentum_unlearn, weight_decay=opt.wd_unlearn)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.epochs_unlearn)
    for _ in tqdm(range(opt.epochs_unlearn)):
        for (img_ret, lab_ret), (img_fgt, lab_fgt) in zip(retain, forget):
            img_ret, lab_ret, img_fgt, lab_fgt = img_ret.to(opt.device), lab_ret.to(opt.device), img_fgt.to(opt.device), lab_fgt.to(opt.device)
            optimizer.zero_grad()

            logits_fgt = bbone(img_fgt)
            outputs_fgt = fc(logits_fgt)
            loss_fgt = torch.nn.functional.cross_entropy(outputs_fgt,lab_fgt) *lambda_1

            logits_ret = bbone(img_ret)
            outputs_ret = fc(logits_ret)
            loss_ret = torch.nn.functional.cross_entropy(outputs_ret,lab_ret) * lambda_2

            loss_logits = torch.nn.KLDivLoss(reduce="batchnorm")(logits_ret.mean(0), logits_fgt.mean(0)) * lambda_3

            loss =  loss_fgt + loss_ret + loss_logits
            #print("LOSS FGT: ", loss_fgt.item(), "LOSS RET: ", loss_ret.item(), "LOSS LOGITS: ", loss_logits.item())
            loss.backward()
            optimizer.step()
        scheduler.step()

    net.eval()
    return net

def fine_tune2(net, retain):
    # split model to get intermediate features
    bbone = net[:-1]
    fc = net[-1]
    bbone.train(), fc.train()
    params=[]
    for idx in range(len(bbone)):
        params.extend(bbone[idx].parameters())
    params.extend(fc.parameters())

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(params, lr=opt.lr_fine_tune, momentum=opt.momentum_fine_tune, weight_decay=opt.wd_fine_tune)
    epochs = opt.epochs_fine_tune
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for _ in tqdm(range(epochs)):
        for inputs, targets in retain:
            inputs, targets = inputs[:].to(opt.device), targets[:].to(opt.device)
            sampled_inputs, sampled_targets=inputs, targets
            
            # fixes a bug when batch size is 1
            if sampled_inputs.shape[0] == 1:
                sampled_inputs = inputs
                sampled_targets = targets
            
            optimizer.zero_grad()
            outputs_ft = fc(bbone(sampled_inputs))
            loss = criterion(outputs_ft, sampled_targets)
            loss.backward()
            optimizer.step()
        scheduler.step()

    # merge model to get intermediate features
    my_net = merge_network2(bbone, fc)
    my_net.eval()

    return my_net




def fine_tune(ft_model, retain_loader, layer_idx):
    """Fine-tune the model on the retain set"""

    # split model to get intermediate features
    bbone = ft_model[:-1]
    fc = ft_model[-1]

    # freeze all layers except the first layer
    for param in bbone.parameters():
        param.requires_grad = True
    fc.requires_grad=True

    for param in bbone[layer_idx].parameters():
        param.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    params=[]
    for idx in range(len(bbone)):
        if idx!=layer_idx:
            params.extend(bbone[idx].parameters())
    params.extend(fc.parameters())
    optimizer = optim.SGD(params, lr=opt.lr_fine_tune, momentum=opt.momentum_fine_tune, weight_decay=opt.wd_fine_tune)
    epochs = opt.epochs_fine_tune
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    bbone.train()
    fc.train()

    for _ in tqdm(range(epochs)):
        for inputs, targets in retain_loader:
            inputs, targets = inputs.to(opt.device), targets.to(opt.device)
            sampled_inputs, sampled_targets=inputs[:int(inputs.shape[0]*0.2)], targets[:int(inputs.shape[0]*0.2)]
            
            # fixes a bug when batch size is 1
            if sampled_inputs.shape[0] == 1:
                sampled_inputs = inputs
                sampled_targets = targets
            
            optimizer.zero_grad()
            outputs_ft = fc(bbone(sampled_inputs))
            loss = criterion(outputs_ft, sampled_targets)
            loss.backward()
            optimizer.step()
        scheduler.step()

    # merge model to get intermediate features
    my_net = merge_network2(bbone, fc)
    my_net.eval()

    return my_net


