import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from opts import OPT as opt
from utils import accuracy


def pairwise_cos_dist(x, y):
    """Compute pairwise cosine distance between two tensors"""
    x_norm = torch.norm(x, dim=1).unsqueeze(1)
    y_norm = torch.norm(y, dim=1).unsqueeze(1)
    x = x / x_norm
    y = y / y_norm
    return 1 - torch.mm(x, y.transpose(0, 1))


def unlearning(net, retain, forget,target_accuracy=0.76, opt=opt):
    """calcola embeddings"""
    #lambda1 fgt
    #lambda2 retain


    bbone = torch.nn.Sequential(*(list(net.children())[:-1] + [nn.Flatten()]))
    if opt.model == 'AllCNN':
        fc = net.classifier
    else:
        fc=net.fc
    
    bbone.eval()

    # embeddings of retain set
    with torch.no_grad():
        ret_embs=[]
        labs=[]
        for img_ret, lab_ret in retain:
            img_ret, lab_ret = img_ret.to(opt.device), lab_ret.to(opt.device)
            logits_ret = bbone(img_ret)
            ret_embs.append(logits_ret)
            labs.append(lab_ret)
        ret_embs=torch.cat(ret_embs)
        labs=torch.cat(labs)
    

    # compute centroids from embeddings
    centroids=[]
    for i in range(opt.num_classes):
        if type(opt.class_to_be_removed) is tuple:
            if not i in opt.class_to_be_removed:
                centroids.append(ret_embs[labs==i].mean(0))
        else:
            if i!=opt.class_to_be_removed:
                centroids.append(ret_embs[labs==i].mean(0))
    centroids=torch.stack(centroids)


    bbone.train(), fc.train()
    #optimizer = optim.SGD(net.parameters(), lr=opt.lr_unlearn, momentum=opt.momentum_unlearn, weight_decay=opt.wd_unlearn)
    optimizer = optim.Adam(net.parameters(), lr=opt.lr_unlearn, weight_decay=opt.wd_unlearn)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.epochs_unlearn)

    init = True
    flag_exit = False
    all_closest_centroids = []
    for _ in tqdm(range(opt.epochs_unlearn)):
        for n_batch, ((img_ret, lab_ret), (img_fgt, lab_fgt)) in enumerate(zip(retain, forget)):
            img_ret, lab_ret, img_fgt, lab_fgt = img_ret.to(opt.device), lab_ret.to(opt.device), img_fgt.to(opt.device), lab_fgt.to(opt.device)
            optimizer.zero_grad()

            logits_fgt = bbone(img_fgt)

            # compute pairwise cosine distance between embeddings and centroids
            dists = pairwise_cos_dist(logits_fgt, centroids)


            # pick the closest centroid that has class different from lab_fgt (only first time)
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
            # outputs_fgt = fc(logits_fgt)

            logits_ret = bbone(img_ret)
            outputs_ret = fc(logits_ret)
            loss_ret = torch.nn.functional.cross_entropy(outputs_ret, lab_ret) * opt.lambda_2
            #print(torch.nn.functional.cross_entropy(outputs_ret, lab_ret))

            loss =  loss_fgt + loss_ret
            print(f"LOSS FGT: {loss_fgt.item():.4f}  -  LOSS RET: {loss_ret.item():.4f}")
            loss.backward()
            optimizer.step()

        # evaluate accuracy on forget set every batch
        with torch.no_grad():
            net.eval()
            curr_acc = accuracy(net, forget)
            tr_acc = accuracy(net, retain)
            net.train()
            print(f"ACCURACY FORGET SET: {curr_acc:.3f}, target is {target_accuracy:.3f}")
            print(f"ACCURACY retain SET: {tr_acc:.3f}")
            if curr_acc < target_accuracy:
                flag_exit = True

        if flag_exit:
            break

        init = False
        #scheduler.step()


    net.eval()
    return net


def unlearning_VGG_rem1subj(net, retain, forget):
    """calcola embeddings"""
    lambda_1, lambda_2 = .1,1   # 1, 0.1
    #lambda1 fgt
    #lambda2 retain
    bbone = torch.nn.Sequential(*(list(net.children())[:-1] + [nn.Flatten()]))
    if opt.model == 'AllCNN':
        fc = net.classifier
    else:
        fc=net.fc
    
    bbone.eval()

    # embeddings of retain set
    with torch.no_grad():
        ret_embs=[]
        labs=[]
        for img_ret, lab_ret in retain:
            img_ret, lab_ret = img_ret.to(opt.device), lab_ret.to(opt.device)
            logits_ret = bbone(img_ret)
            ret_embs.append(logits_ret)
            labs.append(lab_ret)
        ret_embs=torch.cat(ret_embs)
        labs=torch.cat(labs)
    

    # compute centroids from embeddings
    centroids=[]
    for i in range(opt.num_classes):
        if i!=opt.class_to_be_removed:
            centroids.append(ret_embs[labs==i].mean(0))
    centroids=torch.stack(centroids)
    

    bbone.train(), fc.train()
    #optimizer = optim.SGD(net.parameters(), lr=opt.lr_unlearn, momentum=opt.momentum_unlearn, weight_decay=opt.wd_unlearn)
    optimizer = optim.Adam(net.parameters(), lr=opt.lr_unlearn, weight_decay=opt.wd_unlearn)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.epochs_unlearn)

    init = True
    flag_exit = False
    all_closest_centroids = []
    for _ in tqdm(range(opt.epochs_unlearn)):
        for n_batch, ((img_ret, lab_ret), (img_fgt, lab_fgt)) in enumerate(zip(retain, forget)):
            img_ret, lab_ret, img_fgt, lab_fgt = img_ret.to(opt.device), lab_ret.to(opt.device), img_fgt.to(opt.device), lab_fgt.to(opt.device)
            optimizer.zero_grad()

            logits_fgt = bbone(img_fgt)

            # compute pairwise cosine distance between embeddings and centroids
            dists = pairwise_cos_dist(logits_fgt, centroids)


            # pick the closest centroid that has class different from lab_fgt (only first time)
            if init:
                closest_centroids = torch.argsort(dists, dim=1)[:,0]
                # tmp = closest_centroids[:, 0]
                # closest_centroids = torch.where(tmp == lab_fgt, closest_centroids[:, 1], tmp)
                all_closest_centroids.append(closest_centroids)
                closest_centroids = all_closest_centroids[-1]
            else:
                closest_centroids = all_closest_centroids[n_batch]

            dists = dists[torch.arange(dists.shape[0]), closest_centroids[:dists.shape[0]]]
            loss_fgt = torch.mean(dists) * lambda_1
            # outputs_fgt = fc(logits_fgt)

            logits_ret = bbone(img_ret)
            outputs_ret = fc(logits_ret)
            loss_ret = torch.nn.functional.cross_entropy(outputs_ret, lab_ret) * lambda_2
            #print(torch.nn.functional.cross_entropy(outputs_ret, lab_ret))

            loss =   loss_ret+loss_fgt
            print(f"LOSS FGT: {loss_fgt.item():.4f}  -  LOSS RET: {loss_ret.item():.4f}")
            loss.backward()
            optimizer.step()
            

        #evaluate accuracy on forget 
        with torch.no_grad():
            net.eval()
            curr_acc = accuracy(net, forget)
            tr_acc = accuracy(net,retain)
            net.train()
            print(f"ACCURACY FORGET SET: {curr_acc:.3f}")
            print(f"ACCURACY retain SET: {tr_acc:.3f}")

            if curr_acc < 0.01:
                flag_exit = True

        if flag_exit:
            break


        init = False
        #scheduler.step()


    net.eval()
    return net