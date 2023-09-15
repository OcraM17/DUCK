import numpy as np


classes = np.arange(10)
occupancy = np.asarray([14622, 1688, 1130, 529, 643, 932, 2333, 1553, 689, 436])

print(f'N is : {occupancy.sum()}')
for i in range(10):
    print(f'class {i} percentage {occupancy[i]/occupancy.sum()}')