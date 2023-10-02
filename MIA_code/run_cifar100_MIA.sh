#!/bin/bash

for i in {2..20}
do
   echo "Run number: $i"
   python train_MIAmodels_cifar100.py --run $i
done
