import os

# Training Hyperparameters
NUM_CLASSES         = 200
BATCH_SIZE          = 2048
VAL_EVERY_N_EPOCH   = 1

# NUM_EPOCHS          = 50
NUM_EPOCHS          = 20
# OPTIMIZER_PARAMS    = {'type': 'SGD', 'lr': 0.005, 'momentum': 0.9}
OPTIMIZER_PARAMS    = {'type': 'Adam', 'lr': 0.001, 'betas': (0.9, 0.999), 'eps': 1e-8, 'weight_decay': 0}
# OPTIMIZER_PARAMS    = {'type': 'AdamW', 'lr': 0.001, 'betas': (0.9, 0.999), 'eps': 1e-8, 'weight_decay': 0}
# SCHEDULER_PARAMS    = {'type': 'MultiStepLR', 'milestones': [30, 35], 'gamma': 0.2}
# SCHEDULER_PARAMS    = {'type': 'CosineAnnealingLR', 'T_max': NUM_EPOCHS, 'eta_min': 1e-6}
SCHEDULER_PARAMS    = {'type': 'OneCycleLR', 'max_lr': 0.01, 'pct_start': 0.3, 'div_factor': 25.0, 'final_div_factor': 10000.0, 'total_steps': int(100000//BATCH_SIZE)*NUM_EPOCHS}


# Dataaset
DATASET_ROOT_PATH   = 'datasets/'
NUM_WORKERS         = 8

# Augmentation
IMAGE_ROTATION      = 20
IMAGE_FLIP_PROB     = 0.5
IMAGE_NUM_CROPS     = 64
IMAGE_PAD_CROPS     = 4
IMAGE_MEAN          = [0.4802, 0.4481, 0.3975]
IMAGE_STD           = [0.2302, 0.2265, 0.2262]

# Network
# MODEL_NAME          = 'resnet18'
# MODEL_NAME          = 'MyNetwork'
# MODEL_NAME          = 'MyAlexNet'
# MODEL_NAME          = 'MyConvNeXtV2'
# MODEL_NAME          = 'MyEfficientNetL2'
MODEL_NAME          = 'MyEfficientNetL2_dropout'


# Compute related
ACCELERATOR         = 'gpu'
# ACCELERATOR         = 'cpu'
DEVICES             = [0]
PRECISION_STR       = '32-true'

# Logging
WANDB_PROJECT       = 'aue8088-pa1'
WANDB_ENTITY        = os.environ.get('WANDB_ENTITY')
WANDB_SAVE_DIR      = 'wandb/'
WANDB_IMG_LOG_FREQ  = 50
WANDB_NAME          = f'{MODEL_NAME}-B{BATCH_SIZE}-{OPTIMIZER_PARAMS["type"]}'
WANDB_NAME         += f'-{SCHEDULER_PARAMS["type"]}{OPTIMIZER_PARAMS["lr"]:.1E}-Ep{NUM_EPOCHS}'

