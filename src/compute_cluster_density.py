import torch
from opts import OPT as opt
import numpy as np
from utils import get_trained_model
from models.resnet import ModifiedResNet
from sklearn.decomposition import PCA
# Note: There's no direct percentile function in PyTorch, so we sort and select indices
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy
import seaborn as sns
import pandas as pd
from scipy.special import gammaln
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
    log_gamma = gammaln((n / 2)+1)
    denominator = math.exp(log_gamma)
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

    l1, l2, l3, out_fc, y = model.extract_feat(loader)
    out_fc = torch.argmax(out_fc, dim=1).detach().cpu()
    y = y.detach().cpu()
    # if p ==90:
    #     print(mode, torch.histogram(torch.tensor(out_fc[y==class_to_remove], dtype=torch.float32), bins=torch.tensor([i for i in range(11)], dtype=torch.float32)))

    l3 = l3.view(-1, 512)

    l2 = l2.view(10000, -1)
    #outputs = outputs.detach().cpu().numpy()
    outputs = l3.detach().cpu().numpy()
    #outputs = l2.detach().cpu().numpy()
    n_components = 100
    pca = PCA(n_components=n_components)
    outputs = torch.tensor(pca.fit_transform(outputs))
    total_variance = pca.explained_variance_.sum()
    threshold = 0.983
    for i in range(n_components):
        if pca.explained_variance_[:i].sum() / total_variance > threshold:
            print("n_components: ", i)
            break
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
        vol = volume_n_sphere(p2, n_components)
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
    #plt.ylim(10e-7, 1001)
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
    #plt.ylim(10e-7, 1001)
    #remove top and right border
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.yscale("log")
    #legend with no frame
    plt.legend(frameon=False)

    plt.savefig(f"density_vs_percentile.svg")
    plt.savefig(f"density_vs_percentile.png")
    plt.savefig(f"density_vs_percentile.pdf")
    plt.close()
    plt.figure()
    

def get_mean_std(path):
    densities = torch.load(path)
    mean_densities = torch.tensor([sum(densities[p])/len(densities[p]) for p in densities.keys()])
    std_densities = torch.tensor([np.std(densities[p])/np.sqrt(len(densities[p])) for p in densities.keys()])
    return mean_densities, std_densities
def plot_from_file():
    tag = "unnormed"
    densities_forget_original = torch.load(opt.root_folder+"out/"+opt.mode+"/"+opt.dataset+"/dfs/densities_forget_original_"+tag+".pth")
    densities_retain_original = torch.load(opt.root_folder+"out/"+opt.mode+"/"+opt.dataset+"/dfs/densities_retain_original_"+tag+".pth")
    densities_forget_unlearned = torch.load(opt.root_folder+"out/"+opt.mode+"/"+opt.dataset+"/dfs/densities_forget_unlearned_"+tag+".pth")
    densities_retain_unlearned = torch.load(opt.root_folder+"out/"+opt.mode+"/"+opt.dataset+"/dfs/densities_retain_unlearned_"+tag+".pth")
    densities_forget_retrained = torch.load(opt.root_folder+"out/"+opt.mode+"/"+opt.dataset+"/dfs/densities_forget_retrained_"+tag+".pth")
    densities_retain_retrained = torch.load(opt.root_folder+"out/"+opt.mode+"/"+opt.dataset+"/dfs/densities_retain_retrained_"+tag+".pth")
    mean_densities_forget_original, std_densities_forget_original = get_mean_std(opt.root_folder+"out/"+opt.mode+"/"+opt.dataset+"/dfs/densities_retain_original_"+tag+".pth")
    mean_densities_retain_original, std_densities_retain_original = get_mean_std(opt.root_folder+"out/"+opt.mode+"/"+opt.dataset+"/dfs/densities_forget_original_"+tag+".pth")
    mean_densities_forget_unlearned, std_densities_forget_unlearned = get_mean_std(opt.root_folder+"out/"+opt.mode+"/"+opt.dataset+"/dfs/densities_forget_unlearned_"+tag+".pth")
    mean_densities_retain_unlearned, std_densities_retain_unlearned = get_mean_std(opt.root_folder+"out/"+opt.mode+"/"+opt.dataset+"/dfs/densities_retain_unlearned_"+tag+".pth")
    mean_densities_forget_retrained, std_densities_forget_retrained = get_mean_std(opt.root_folder+"out/"+opt.mode+"/"+opt.dataset+"/dfs/densities_forget_retrained_"+tag+".pth")
    mean_densities_retain_retrained, std_densities_retain_retrained = get_mean_std(opt.root_folder+"out/"+opt.mode+"/"+opt.dataset+"/dfs/densities_retain_retrained_"+tag+".pth")

    #print(densities_forget_unlearned)
    keys = torch.load(opt.root_folder+"out/"+opt.mode+"/"+opt.dataset+"/dfs/densities_retain_original.pth").keys()

    for k in [95]:
        print(k)
        print(f"wilcoxon retain unlearned/retrained: {scipy.stats.mannwhitneyu(densities_retain_unlearned[k], densities_retain_retrained[k])}") 
        print(f"wilcoxon forget unlearned/retrained: {scipy.stats.mannwhitneyu(densities_forget_unlearned[k], densities_forget_retrained[k])}")
        print(f"wilcoxon unlearned retain/forget: {scipy.stats.mannwhitneyu(densities_retain_unlearned[k], densities_forget_unlearned[k])}") 
        print(f"wilcoxon retrained retain/forget: {scipy.stats.mannwhitneyu(densities_retain_retrained[k], densities_forget_retrained[k])}")
        print(f"wilcoxon original retain/forget: {scipy.stats.mannwhitneyu(densities_retain_original[k], densities_forget_original[k])}")
    
    '''print("\n\nDensities for each percentile Original:")
    for i,k in enumerate(keys):
        print(f"Percentile {k}:\n\tForget: {mean_densities_forget_original[i]} \\pm {std_densities_forget_original[i]}\n\tRetain: {mean_densities_retain_original[i]} \\pm {std_densities_retain_original[i]}\n")
    print("\n\nDensities for each percentile Unlearned:")
    for i,k in enumerate(keys):
        print(f"Percentile {k}:\n\tForget: {mean_densities_forget_unlearned[i]} \\pm {std_densities_forget_unlearned[i]}\n\tRetain: {mean_densities_retain_unlearned[i]} \\pm {std_densities_retain_unlearned[i]}\n")
    print("\n\nDensities for each percentile Retrained:")
    for i,k in enumerate(keys):
        print(f"Percentile {k}:\n\tForget: {mean_densities_forget_retrained[i]} \\pm {std_densities_forget_retrained[i]}\n\tRetain: {mean_densities_retain_retrained[i]} \\pm {std_densities_retain_retrained[i]}\n")
    '''
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
    #plt.ylim(10e-3, 1e10)
    #remove top and right border
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.yscale("log")
    #legend with no frame
    plt.legend(frameon=False)
    plt.savefig(f"{opt.root_folder}/plots/density_vs_percentile_{tag}.svg")
    plt.savefig(f"{opt.root_folder}/plots/density_vs_percentile_{tag}.png")
    plt.savefig(f"{opt.root_folder}/plots/density_vs_percentile_{tag}.pdf")
    plt.close()
    k = 95
    df = pd.DataFrame({
    'Original-Forget': densities_forget_original[k],
    'Original-Retain': densities_retain_original[k],
    'DUCK-Forget': densities_forget_unlearned[k],
    'DUCK-Retain': densities_retain_unlearned[k],
    'Retrained-Forget': densities_forget_retrained[k],
    'Retrained-Retain': densities_retain_retrained[k]
})

    # Convert the DataFrame to a 'long-form' or 'tidy' format
    df_melted = df.melt(var_name='Case', value_name='Density')

    # Splitting 'Case' into 'Category' and 'Memory' using a '-' as the separator
    df_melted['Category'], df_melted['Case'] = zip(*df_melted['Case'].apply(lambda x: x.split('-')))

    # Create the box plot
    plt.figure(figsize=(3.5, 2))
    sns.set_palette('Pastel2')

    ax = sns.boxplot(x='Category', y='Density', hue='Case', data=df_melted, showfliers=False, medianprops={"color": "firebrick", "linewidth": 1.}, linewidth=1, width=0.5, fliersize=0.5, palette=["powderblue", "floralwhite"])
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    #set bigger legend
    plt.legend(frameon=False, loc="lower left", prop= {'size': 8})

    ax.set_xlabel('Category', fontsize=8, labelpad=20)  # Change the font size for x-axis label
    ax.set_ylabel('Density', fontsize=8, labelpad=20)   # Change the font size for y-axis label
    ax.tick_params(axis='x', labelsize=7)  # Change the font size for x-axis ticks
    ax.tick_params(axis='y', labelsize=7)  # Change the font size for y-axis ticks
    #plt.ylim(1e-5, 1e-2)

    plt.yscale("log")
    font = {'size'   : 8}

    mpl.rc('font', **font)
    mpl.rcParams['axes.spines.right'] = False
    mpl.rcParams['axes.spines.top'] = False
    #plt.title('Mean Densities with Standard Deviation', fontsize=20)
    plt.savefig(f"{opt.root_folder}/plots/boxplot_{tag}.png")
    plt.savefig(f"{opt.root_folder}/plots/boxplot_{tag}.svg")
    plt.savefig(f"{opt.root_folder}/plots/boxplot_{tag}.pdf")
                                                    
#test function (main)

if __name__ == '__main__':
    #load model
    plot_from_file()