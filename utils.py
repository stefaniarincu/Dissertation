import os
import random
import numpy as np
import datetime
import torch
from torch.nn import functional as F

''' ============================================ REPRODUCIBILITY ============================================ '''

# Function that sets constant seed for reproducibility
def seed_all(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


''' =============================================== LOGGING =============================================== '''

# Function that ensures that a log file exists and writes the start datetime to it
def create_log_file(log_path):
    # If it exists write the start datetime else create the file and write the start datetime
    if os.path.exists(log_path):
        print('Log file already exists')
    else:
        with open(log_path, 'w') as f:
            f.write('\n')
    
    # Save the start datetime to the log file
    start_datetime = str(datetime.datetime.now())
    print_and_save(log_path, start_datetime)

# Function that prints a message and also saves it to a specified file
def print_and_save(file_path, text):
    print(text)
    with open(file_path, 'a') as file:
        file.write(text)
        file.write('\n')


''' ===================================== CHECKPOINT TO RESUME TRAINING ===================================== '''

# Function that saves a checkpoint for resuming training later if needed
def save_resume_checkpoint(model, epoch, optimizer, scheduler, best_validation_metric, num_epochs_no_improvement, path):
    torch.save({
        'model_state': model.state_dict(),
        'epoch': epoch,
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict(),
        'best_validation_metric': best_validation_metric,
        'num_epochs_no_improvement': num_epochs_no_improvement,
        'rng_state': {
            'python': random.getstate(),
            'numpy': np.random.get_state(),
            'torch': torch.get_rng_state().cpu(),
            'cuda': [state.cpu() for state in torch.cuda.get_rng_state_all()]
        }
    }, path)

# Function that loads a checkpoint and restores the model, optimizer, scheduler states, as well as RNG states for reproducibility
def load_resume_checkpoint(model, optimizer, scheduler, path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    scheduler.load_state_dict(checkpoint['scheduler_state'])
    best_validation_metric = checkpoint['best_validation_metric']
    num_epochs_no_improvement = checkpoint['num_epochs_no_improvement']
    epoch = checkpoint['epoch'] + 1

    # Restore RNG states
    random.setstate(checkpoint['rng_state']['python'])
    np.random.set_state(checkpoint['rng_state']['numpy'])
    torch_state = checkpoint['rng_state']['torch']
    if not isinstance(torch_state, torch.Tensor):
        torch_state = torch.tensor(torch_state, dtype=torch.uint8)
    else:
        torch_state = torch_state.to(dtype=torch.uint8)
    torch.set_rng_state(torch_state.cpu())

    cuda_states = checkpoint['rng_state']['cuda']
    fixed_cuda_states = []
    for state in cuda_states:
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.uint8)
        else:
            state = state.to(dtype=torch.uint8)
        fixed_cuda_states.append(state.cpu())
    torch.cuda.set_rng_state_all(fixed_cuda_states)

    return epoch, best_validation_metric, num_epochs_no_improvement


''' ======================================= FREEZE MODEL PARAMETERS ======================================= '''

# Function that freezes the parameters of a given model and sets it to evaluation mode
def freeze_model_parameters(model):
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    return model


''' ============================== CREATE OPTIMIZER FROM TRAINABLE PARAMETERS ============================== '''

# Function that creates the optimizer with just the trainable parameters
def create_optimizer(model, learning_rate):
    trainable_parameters = filter(lambda p: p.requires_grad, model.parameters())
    return torch.optim.Adam(trainable_parameters, lr=learning_rate)


''' ===================================== LOAD DATASET-SPECIFIC MODELS  ===================================== '''

# Function that loads dataset specific models from specified checkpoints and returns a dictionary that maps dataset ids to the corresponding model
def load_dataset_specific_models(checkpoints, dataset_specific_models, device):
    for dataset_id, checkpoint_path in checkpoints.items():
        dataset_specific_models[dataset_id].load_state_dict(torch.load(checkpoint_path, map_location=device))
        dataset_specific_models[dataset_id].to(device)

        dataset_specific_models[dataset_id] = freeze_model_parameters(dataset_specific_models[dataset_id])
    return dataset_specific_models


''' ============================== DETERMINE WEIGHTS FOR DATASET-SPECIFIC MODELS ============================== '''

# Determine the weights for each dataset specific model based on the specified mode and the dataset ids of the samples in the batch
def compute_dataset_specific_weights(dataset_ids, weighting_mode, num_dataset_specific_models):
    # one hot encoding = > 1 for the dataset specific model corresponding to the dataset and 0 for the others
    if weighting_mode == 0: 
        return F.one_hot(dataset_ids, num_classes=num_dataset_specific_models).float().to(dataset_ids.device)
    # uniform weights = > 1/num_dataset_specific_models for all dataset specific models
    elif weighting_mode == 1: 
        return torch.full((dataset_ids.shape[0], num_dataset_specific_models), 1.0 / num_dataset_specific_models, device=dataset_ids.device, dtype=torch.float32)
    # biased weights = > 0.5 for the dataset specific model corresponding to the dataset and 0.25 for the others
    elif weighting_mode == 2: 
        weights = torch.full((dataset_ids.shape[0], num_dataset_specific_models), 0.25, device=dataset_ids.device, dtype=torch.float32)
        return weights.scatter_(1, dataset_ids.view(-1, 1), 0.5)
    
