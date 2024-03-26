import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import random
import pickle as pk


#open dfs from out/CR/CIFAR100 and extract retain_test_accuracy

seeds = [i for i in range(4)]+[42]
path = 'out/CR/cifar100/dfs'
accuracies = {1:[], 10:[], 20:[], 30:[], 40:[], 50:[], 60:[], 70:[], 80:[], 90:[], 98:[]}
accuracies_Finetuning = {1:[], 10:[], 20:[], 30:[], 40:[], 50:[], 60:[], 70:[], 80:[], 90:[], 98:[]}
for s in seeds:
    random.seed(s)
    class_range = [i for i in range(100)]
    random.shuffle(class_range)
    class_to_remove = [class_range[:j] for j in [1]+ [z*10 for z in range(1,10)]+[98]]
    for classes in class_to_remove:
        df = pd.read_csv(f"{path}/DUCK_seed_{s}_class_{classes[0]}_{classes[-1]}.csv")
        accuracies[len(classes)].append(df['retain_test_accuracy'].mean())
        
        df = pd.read_csv(f"{path}/FineTuning_seed_{s}_class_{classes[0]}_{classes[-1]}.csv")
        accuracies_Finetuning[len(classes)].append(df['retain_test_accuracy'].mean())
#print(accuracies)
means = []
stds = []
means_finetuning = []
stds_finetuning = []
for classes in class_to_remove:
    means.append(np.mean(accuracies[len(classes)]))
    stds.append(np.std(accuracies[len(classes)]))
    means_finetuning.append(np.mean(accuracies_Finetuning[len(classes)]))
    stds_finetuning.append(np.std(accuracies_Finetuning[len(classes)]))
means = np.array(means)*100
stds = np.array(stds)*100
means_finetuning = np.array(means_finetuning)*100
stds_finetuning = np.array(stds_finetuning)*100
print("means",means)
print("stds",stds)
print("means_finetuning",means_finetuning)
print("stds_finetuning",stds_finetuning)
# df = pd.read_csv(f"{path}/Finetuning_multiple_class.csv", header = None)
# #take df second column
# means_finetuning = df.iloc[:,1]
# df = pd.read_csv(f"{path}/Finetuning_multiple_class_std.csv", header=None)
# #take df second column
# stds_finetuning = df.iloc[:,1] - means_finetuning
xs = [len(classes) for classes in class_to_remove]
#invert mens sorting
#means_finetuning = np.flip(np.array(means_finetuning))
#stds_finetuning = np.flip(np.array(stds_finetuning))

print("means_finetuning",means_finetuning)
print("stds_finetuning",stds_finetuning)
#plot
fig, ax = plt.subplots(figsize=(3.5, 2), dpi=500)
plt.plot(xs, means_finetuning, label = 'Finetuning', color = "purple")
plt.fill_between(xs, means_finetuning - stds_finetuning, means_finetuning + stds_finetuning, alpha=0.2, color = "purple")
plt.plot(xs,means, label = 'DUCK', color = "orange")
plt.fill_between(xs, means - stds, means + stds, alpha=0.2, color = "orange")
plt.xlabel('Number of classes removed', fontsize=8)
plt.rcParams["text.usetex"] = True
plt.ylabel('Accuracy (%)', fontsize=8)
plt.ylim(70, 100)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
ax.tick_params(axis='x', labelsize=7)  # Change the font size for x-axis ticks
ax.tick_params(axis='y', labelsize=7)  # Change the font size for y-axis ticks
plt.legend(frameon=False)
plt.savefig(f'plots/Finetuning_multiple_class.png', bbox_inches='tight')  # Save the plot