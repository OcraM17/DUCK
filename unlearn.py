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
        

def unlearning(net, layer_idx, retain, forget, validation):
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

    return ft_model


