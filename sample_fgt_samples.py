import numpy as np
import torchvision

data_path = '~/data'
np.random.seed(42)

train_set_cifar10 = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True)
train_set_cifar100 = torchvision.datasets.CIFAR100(root=data_path, train=True, download=True)

for num in [1000,5000]:
    forget_idx_cifar10 = np.random.choice(len(train_set_cifar10.targets), size=num, replace=False)
    forget_idx_cifar100 = np.random.choice(len(train_set_cifar100.targets), size=num, replace=False)

    #save the indices into a txt file
    np.savetxt(f'forget_idx_{num}_cifar10.txt', forget_idx_cifar10.astype(np.int64))
    np.savetxt(f'forget_idx_{num}_cifar100.txt', forget_idx_cifar100.astype(np.int64))


