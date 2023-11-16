#!/bin/bash

for i in {2..5}
do
    python3 training_cifar_multiple_removal.py --seed $i
done