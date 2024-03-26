import os
import random
seeds = [i for i in range(5)]+[42]

for s in seeds:
    #shuffle class range
    random.seed(s)
    class_range = [i for i in range(100)]
    random.shuffle(class_range)
    class_to_remove = str([class_range[:j] for j in [10]])#+ [z*10 for z in range(1,10)]+[98]])
    #lambda_1 = 1.5/(len())
    #print(class_to_remove)
    #DUCK
    #os.system(f'python src/main_def.py --seed {s} --class_to_remove "{class_to_remove}" --run_unlearn --lambda_1 1.5 --lambda_2 1.5 --lr 0.001 --bsize 1024 --epochs 25 --dataset cifar100 --mode CR --cuda 0 ')
    #os.system(f'python src/main_def.py --seed {s} --class_to_remove "{class_to_remove}" --run_unlearn --load_unlearned_model --lambda_1 0.5 --lambda_2 1.5 --lr 0.0002 --bsize 1024 --epochs 30 --dataset tinyImagenet --mode CR --cuda 0 --temperature 3')
    os.system(f'python src/main_def.py --seed {s} --class_to_remove "{class_to_remove}" --run_original --lambda_1 0.5 --lambda_2 1.5 --lr 0.001 --bsize 256 --epochs 10 --dataset tinyImagenet --mode CR --cuda 0 --wd 5e-5')

    #Finetuning
    #os.system(f'python src/main_def.py --method FineTuning --seed {s} --class_to_remove "{class_to_remove}" --run_unlearn --lr 0.01 --bsize 32 --epochs 40 --dataset tinyImagenet --mode CR --scheduler 25 35 --cuda 1 --save_model --save_df')
