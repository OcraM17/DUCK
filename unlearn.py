import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


def split_network(net):
    # split model to get intermediate features
    bbone = nn.Sequential(
        net.conv1,
        net.bn1,
        net.relu,
        net.maxpool,
        net.layer1,
        net.layer2,
        net.layer3,
        net.layer4,
        net.avgpool,
        nn.Flatten()
    )

    # split model to get last fully connected layer
    fc = net.fc

    return bbone, fc


def merge_network(bbone, fc):
    # merge model to get intermediate features
    net = nn.Sequential(
        bbone[0],
        bbone[1],
        bbone[2],
        bbone[3],
        bbone[4],
        bbone[5],
        bbone[6],
        bbone[7],
        bbone[8],
        bbone[9],
        fc
    )

    return net


def unlearning(DEVICE, net, layer_idx, retain, forget, validation):
    """Unlearning by diocan"""

    # split model to get intermediate features
    bbone, fc = split_network(net)

    # freeze all layers except the first layer
    for param in bbone.parameters():
        param.requires_grad = False

    for param in bbone[layer_idx].parameters():
        param.requires_grad = True

    
    epochs = 5

    #criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(bbone[layer_idx].parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    bbone.train()
    fc.train()

    for _ in range(epochs):
        for inputs, targets in forget:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            outputs_ft = bbone(inputs)
            
            # compare outputs_ft with a random tensor
            outputs_rand = torch.randn_like(outputs_ft)
            loss = torch.nn.functional.mse_loss(outputs_ft, outputs_rand)
            loss.backward()
            optimizer.step()
        scheduler.step()


    # merge model to get intermediate features
    my_net = merge_network(bbone, fc)
    my_net.eval()

    return my_net


def fine_tune(DEVICE, ft_model, retain_loader, layer_idx):
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

    optimizer = optim.SGD(params, lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
    bbone.train()
    fc.train()

    for _ in range(4):
        for inputs, targets in retain_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            inputs, targets=inputs[:int(inputs.shape[0]*0.2)], targets[:int(inputs.shape[0]*0.2)]
            optimizer.zero_grad()
            outputs_ft = fc(bbone(inputs))
            loss = criterion(outputs_ft, targets)
            loss.backward()
            optimizer.step()
        scheduler.step()

    # merge model to get intermediate features
    my_net = merge_network(bbone, fc)
    my_net.eval()

    return ft_model


