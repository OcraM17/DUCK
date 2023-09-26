#!/bin/bash

for i in {11..20}
do
   echo "Run number: $i"
   python train_MIAmodels_VGG.py --run $i
done
