import os
seeds = [i for i in range(5)]+[42]
class_to_remove = str([[i for i in range(100)][:j] for j in [1]+ [z*10 for z in range(1,10)]+[98]])

for s in seeds:
    os.system(f'python src/main_def.py --seed {s} --class_to_remove "{class_to_remove}" --run_unlearn --lambda_1 1.5 --lambda_2 1.5 --lr 0.001 --bsize 1024 --epochs 25 --dataset cifar100 --mode CR --save_df')