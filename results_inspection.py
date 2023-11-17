import pickle as pkl
import numpy as np
import torch
import matplotlib.pyplot as plt

file = open('/home/jb/Documents/MachineUnlearning/cifar10_original_model.pkl','rb')
lst_orig = pkl.load(file)
out_all_fgt,out_all_ret,logits_all_fgt,logits_all_ret,cls_fgt,cls_ret = lst_orig[0],lst_orig[1],lst_orig[2],lst_orig[3],lst_orig[4],lst_orig[5]

file = open('/home/jb/Documents/MachineUnlearning/cifar10_unlearned.pkl','rb')
lst_unlr = pkl.load(file)

out_all_unlr_fgt,out_all_unlr_ret,logits_all_unlr_fgt,logits_all_unlr_ret,cls_unlr_fgt,cls_unlr_ret = lst_unlr[0],lst_unlr[1],lst_unlr[2],lst_unlr[3],lst_unlr[4],lst_unlr[5]
index = torch.argmax(out_all_fgt,dim=1)

for i in range(0,100,10):
    
    buff = out_all_fgt[cls_fgt==i]#torch.nn.functional.softmax(out_all_fgt[index==i],dim=1)
    buff_unlr = out_all_unlr_fgt[cls_unlr_fgt==i]#torch.nn.functional.softmax(out_all_unlr_fgt[index==i],dim=1)

    plt.plot(np.arange(100),buff.mean(0),color='k')
    plt.fill_between(np.arange(100),buff.mean(0)-buff.std(0),buff.mean(0)+buff.std(0),color='k',alpha=0.5)
    
    plt.plot(np.arange(100),buff_unlr.mean(0),color='r')
    plt.fill_between(np.arange(100),buff_unlr.mean(0)-buff_unlr.std(0),buff_unlr.mean(0)+buff_unlr.std(0),color='r',alpha=0.5)

    plt.title(f'Class: {i}')
    plt.savefig(f'./plot/test_plot_fgt_class{i}.png')
    plt.close()

index = torch.argmax(out_all_ret,dim=1)
for i in range(0,100,10):
    buff = out_all_ret[cls_ret==i]#torch.nn.functional.softmax(out_all_ret[index==i],dim=1)
    buff_unlr = out_all_unlr_ret[cls_unlr_ret==i]#torch.nn.functional.softmax(out_all_unlr_ret[index==i],dim=1)
    plt.plot(np.arange(100),buff.mean(0),color='k')
    plt.fill_between(np.arange(100),buff.mean(0)-buff.std(0),buff.mean(0)+buff.std(0),color='k',alpha=0.5)
    
    plt.plot(np.arange(100),buff_unlr.mean(0),color='r')
    plt.fill_between(np.arange(100),buff_unlr.mean(0)-buff_unlr.std(0),buff_unlr.mean(0)+buff_unlr.std(0),color='r',alpha=0.5)

    plt.title(f'Class: {i}')
    plt.savefig(f'./plot/test_plot_ret_class{i}.png')
    plt.close()

# for i in range(5):
#     plt.plot(np.arange(100),out_all_fgt[i,:],color='k')
#     plt.plot(np.arange(100),out_all_unlr_fgt[i,:],color='r')
#     plt.title(f'Class: {torch.argmax(out_all_fgt[i,:])}')
#     plt.savefig(f'test_plot_fgt{i}.png')
#     plt.close()


from sklearn.manifold import TSNE

fig,ax = plt.subplots(ncols=2,nrows=2,figsize=(8,8))
N=1
features = logits_all_fgt.numpy()
features2 = logits_all_ret.numpy()


X =np.concatenate((features[cls_fgt<10][::N,:],features2[cls_ret<10][::N,:]),axis=0)

classes = cls_fgt[cls_fgt<10][::N]
classes2 = cls_ret[cls_ret<10][::N]

mod = TSNE(n_components=2, learning_rate='auto',init='random', perplexity=3)

X_transf = mod.fit_transform(X)
X_emb = X_transf[:features[cls_fgt<10][::N,:].shape[0],:]
X_emb2 = X_transf[features[cls_fgt<10][::N,:].shape[0]:,:]

color = ['k','red','green','gold','pink','violet','orange','aqua','blue','gainsboro']
for i in range(0,10):
    ax[0,0].scatter(X_emb[classes==i,0],X_emb[classes==i,1],c=color[i])
    ax[1,0].scatter(X_emb2[classes2==i,0],X_emb2[classes2==i,1],c=color[i])


features = logits_all_unlr_fgt.numpy()
features2 = logits_all_unlr_ret.numpy()

X =np.concatenate((features[cls_unlr_fgt<10][::N,:],features2[cls_unlr_ret<10][::N,:]),axis=0)
classes = cls_unlr_fgt[cls_unlr_fgt<10][::N]
classes2 = cls_unlr_ret[cls_unlr_ret<10][::N]

mod = TSNE(n_components=2, learning_rate='auto',init='random', perplexity=3)

X_transf = mod.fit_transform(X)

X_emb = X_transf[:features[cls_unlr_fgt<10][::N,:].shape[0],:]
X_emb2 = X_transf[features[cls_unlr_fgt<10][::N,:].shape[0]:,:]

color = ['k','red','green','gold','pink','violet','orange','aqua','blue','gainsboro']
for i in range(0,10):
    ax[0,1].scatter(X_emb[classes==i,0],X_emb[classes==i,1],c=color[i])
    ax[1,1].scatter(X_emb2[classes2==i,0],X_emb2[classes2==i,1],c=color[i])
ax[0,1].set_title('UNlearned fgt')
ax[1,1].set_title('UNlearned ret')
plt.savefig(f'./plot/tsne_fgt_unlr.png')
plt.close()