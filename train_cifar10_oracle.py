from MIA_code.utils import train_model
from opts import OPT as opt
from torchvision.models.resnet import resnet18
from copy import deepcopy
from dsets import get_dsets_remove_class, get_dsets
import torch
import torch.nn as nn
from utils import set_seed
import argparse

if __name__ == "__main__":  

    # Create the parser
    parser = argparse.ArgumentParser(description="This is a description of what this script does")

    # Add the arguments
    parser.add_argument('--run', type=int, default=1)
    parser.add_argument('--test', type=bool, default=False)
    # Execute the parse_args() method
    args = parser.parse_args()
    
    print(f'RUN num: {args.run}')

    # set random seed
    set_seed(opt.seed)

    ##### GET DATA #####
    #if opt.class_to_remove is None:
    #    train_loader, test_loader, train_fgt_loader, train_retain_loader = get_dsets()
    #else:
    all_train_loader,all_test_loader, train_fgt_loader, train_retain_loader, test_fgt_loader, test_retain_loader = get_dsets_remove_class(opt.class_to_remove)

    print(f'training dataloader batches: {len(train_retain_loader)}')
    print(f'test dataloader batches: {len(test_retain_loader)}')

    # Load the pretrained model
    model =resnet18(num_classes=10)#weights=ResNet18_Weights.IMAGENET1K_V1)
    # Change the final layer
    #model.fc = nn.Linear(model.fc.in_features, 100)#nn.Sequential(nn.Dropout(p=0.1),)#0.2

    model = model.to(opt.device)

    # Define the criterion and the optimizer
    criterion = nn.CrossEntropyLoss()#label_smoothing=.06) #weight=class_weights)
    #optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr,weight_decay=opt.wd)
    # Use Cosine Annealing scheduler
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.epochs)
    #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60,120,160], gamma=0.2, last_epoch=-1, verbose=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 250], gamma=0.1)
    # Train the model
    opt.model_weights_name = 'oracle_cifar10_class_rem'
    opt.epochs=200
    train_model(model,optimizer,criterion,scheduler,train_retain_loader,test_retain_loader,opt,args,plot_name=None)