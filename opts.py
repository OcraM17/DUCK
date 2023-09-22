import torch 

class OPT:

    seed = 42
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Model
    model = 'resnet50'#'AllCNN'

    # Data
    data_path = '~/data'
    num_classes = 10
    num_workers = 4
    batch_size = 128
    
    # unlearning params
    class_to_be_removed = 5
    epochs_unlearn = 2000 #best 5
    lr_unlearn = 0.0001#0.0000005 #best 0.001
    wd_unlearn = 0.
    momentum_unlearn = 0.

    # finetuning params
    epochs_fine_tune = 5
    batch_size_FT = 16
    lr_fine_tune = 0.01
    wd_fine_tune = 5e-4
    momentum_fine_tune = 0.9
