import torch
from opts import OPT as opt
import numpy as np
from utils import get_trained_model
from models.resnet import ModifiedResNet
from sklearn.decomposition import PCA
# Note: There's no direct percentile function in PyTorch, so we sort and select indices
import math
import matplotlib.pyplot as plt
import scipy
def volume_n_sphere(radius, n):
    """
    Computes the volume of an n-dimensional sphere with the given radius.
    
    :param radius: Radius of the n-sphere.
    :param n: Dimensionality of the space.
    :return: Volume of the n-sphere.
    """
    # Compute the numerator (pi^(n/2))
    numerator = math.pi ** (n / 2)
    # Compute the denominator (Gamma(n/2 + 1))
    denominator = math.gamma((n / 2) + 1)
    # Compute the volume
    volume = numerator / denominator * (radius ** n)
    return volume
def percentile(t, q):
    k = 1 + round(.01 * float(q) * (t.numel() - 1))
    result = t.view(-1).kthvalue(k).values.item()
    return result

#compute cluster point density at model layer_n output
def density(model, loader, p, class_to_remove, mode="original"):#):
    model.eval()

    # outputs = []
    # y = []
    # with torch.no_grad():
    #     for data in loader:
    #         inputs, labels = data
    #         outputs.append(model.pred_layer_n(inputs, layer_n)) #batchX512
    #         y.append(labels)

    # outputs = torch.cat(outputs)
    # y = torch.cat(y)
    l1, l2, l3, out_fc, y = model.extract_feat(loader)
    out_fc = torch.argmax(out_fc, dim=1).detach().cpu()
    y = y.detach().cpu()
    l3 = l3.view(-1, 512)
    #outputs = outputs.detach().cpu().numpy()
    outputs = l3.detach().cpu().numpy()
    n_components = 9
    pca = PCA(n_components=n_components)
    outputs = torch.tensor(pca.fit_transform(outputs))
    #print(pca.explained_variance_, outputs.shape)

    y = y.detach().cpu()
    uniques_y = torch.unique(y)
    d = {}
    vols = {}
    centroid = torch.mean(outputs[y==out_fc], 0)
    dist = torch.norm(outputs[y==out_fc]-centroid, dim=1)
    p_all = percentile(dist, p)

    for y_i in uniques_y:
        if y_i==class_to_remove:  
            if mode=="original":
                output_i = outputs[torch.logical_and(y == out_fc, y == y_i)] 
            else:
                output_i = outputs[torch.logical_and(y != out_fc, y == y_i)]
        else:
            output_i = outputs[torch.logical_and(y == out_fc, y == y_i)]
            
        centroid = torch.mean(output_i, 0)

        #find distance
        dist = torch.norm(output_i-centroid, dim=1)
        p2 = percentile(dist, p)

        rout = output_i[dist<=p2]
        vol = volume_n_sphere(p2/p_all, n_components)
        vols[y_i.item()] = vol
        d[y_i.item()] = rout.shape[0]/(vol)

        
    # max_vol = max(vols.values())

    # for k, v in d.items():
    #     d[k] = d[k]*max_vol
    return d, vols
def mean_densities_per_percentile(densities_per_class, keys, tag = ""):
    densities_retain = {k:[] for k in keys}
    densities_forget = {k:[] for k in keys}

    for i in range(len(densities_per_class)):
        for p in densities_per_class[i]:
            densities_retain[p].append(np.mean([densities_per_class[i][p][j] for j in range(10) if i != j])) 
            densities_forget[p].append(densities_per_class[i][p][i])
    torch.save(densities_retain, opt.root_folder+"out/"+opt.mode+"/"+opt.dataset+"/dfs/densities_retain_"+tag+".pth")
    torch.save(densities_forget, opt.root_folder+"out/"+opt.mode+"/"+opt.dataset+"/dfs/densities_forget_"+tag+".pth")
    print(f"wilcoxon {tag}: {scipy.stats.wilcoxon(list(densities_forget.values()), list(densities_retain.values()))}") 

    mean_densities_forget = torch.tensor([sum(densities_forget[p])/len(densities_forget[p]) for p in densities_forget.keys()])
    std_densities_forget = torch.tensor([np.std(densities_forget[p])/np.sqrt(len(densities_forget[p])) for p in densities_forget.keys()])

    mean_densities_retain = torch.tensor([sum(densities_retain[p])/len(densities_retain[p]) for p in densities_retain.keys()])
    std_densities_retain = torch.tensor([np.std(densities_retain[p])/np.sqrt(len(densities_retain[p])) for p in densities_retain.keys()])

    return mean_densities_forget, std_densities_forget, mean_densities_retain, std_densities_retain

def plot_densities(mean_densities_forget, std_densities_forget, mean_densities_retain, std_densities_retain, keys, tag = ""):
    plt.figure()
    plt.title("Density vs. Percentile")
    plt.xlabel("Percentile")
    plt.ylabel("Density")
    plt.plot(keys, mean_densities_forget, label="Forget", color="firebrick")
    plt.fill_between(keys, mean_densities_forget-std_densities_forget, mean_densities_forget+std_densities_forget, alpha=0.2, color="firebrick")
    plt.plot(keys, mean_densities_retain, label="Retain", color="darkcyan")
    plt.fill_between(keys, mean_densities_retain-std_densities_retain, mean_densities_retain+std_densities_retain, alpha=0.2, color="darkcyan")
    #set ticks every 5 percent  
    ps = range(70//5, 100//5, 5)
    plt.xticks([i*5 for i in ps], [i*5 for i in ps])
    plt.ylim(10e-7, 1001)
    #remove top and right border
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.yscale("log")
    #legend with no frame
    plt.legend(frameon=False)

    plt.savefig(f"density_vs_percentile_{tag}.svg")
    plt.savefig(f"density_vs_percentile_{tag}.png")
    plt.savefig(f"density_vs_percentile_{tag}.pdf")
def plot_densities_all(mean_densities_forget_original, std_densities_forget_original, mean_densities_retain_original, std_densities_retain_original, mean_densities_forget_unlearned, std_densities_forget_unlearned, mean_densities_retain_unlearned, std_densities_retain_unlearned, mean_densities_forget_retrained, std_densities_forget_retrained, mean_densities_retain_retrained, std_densities_retain_retrained, keys):
    plt.figure()
    plt.title("Density vs. Percentile")
    plt.xlabel("Percentile")
    plt.ylabel("Density")
    plt.plot(keys, mean_densities_forget_original, label="Forget_Original")
    plt.fill_between(keys, mean_densities_forget_original-std_densities_forget_original, mean_densities_forget_original+std_densities_forget_original, alpha=0.2)
    plt.plot(keys, mean_densities_retain_original, label="Retain Original")
    plt.fill_between(keys, mean_densities_retain_original-std_densities_retain_original, mean_densities_retain_original+std_densities_retain_original, alpha=0.2)
    plt.plot(keys, mean_densities_forget_unlearned, label="Forget Unlearned")
    plt.fill_between(keys, mean_densities_forget_unlearned-std_densities_forget_unlearned, mean_densities_forget_unlearned+std_densities_forget_unlearned, alpha=0.2)
    plt.plot(keys, mean_densities_retain_unlearned, label="Retain Unlearned")
    plt.fill_between(keys, mean_densities_retain_unlearned-std_densities_retain_unlearned, mean_densities_retain_unlearned+std_densities_retain_unlearned, alpha=0.2)
    plt.plot(keys, mean_densities_forget_retrained, label="Forget Retrained")
    plt.fill_between(keys, mean_densities_forget_retrained-std_densities_forget_retrained, mean_densities_forget_retrained+std_densities_forget_retrained, alpha=0.2)
    plt.plot(keys, mean_densities_retain_retrained, label="Retain Retrained")
    plt.fill_between(keys, mean_densities_retain_retrained-std_densities_retain_retrained, mean_densities_retain_retrained+std_densities_retain_retrained, alpha=0.2)
    #set ticks every 5 percent  
    ps = range(70//5, 100//5, 5)
    plt.xticks([i*5 for i in ps], [i*5 for i in ps])
    plt.ylim(10e-7, 1001)
    #remove top and right border
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.yscale("log")
    #legend with no frame
    plt.legend(frameon=False)

    plt.savefig(f"density_vs_percentile.svg")
    plt.savefig(f"density_vs_percentile.png")
    plt.savefig(f"density_vs_percentile.pdf")


#test function (main)

if __name__ == '__main__':
    #load model
    original_model = get_trained_model()
    model = ModifiedResNet(original_model)
    density(model,90)