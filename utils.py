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

'''def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)'''

''' =============================================== LOGGING =============================================== '''

# Function that ensures a log file exists and writes the start datetime to it
def create_log_file(log_path):
    # If it exists print a message
    if os.path.exists(log_path):
        print('Log file already exists')
    # Otherwise, create an empty log file
    else:
        with open(log_path, 'w') as f:
            f.write('\n')
    
    # Save the start datetime to the log file
    start_datetime = str(datetime.datetime.now())
    print_and_save(log_path, start_datetime)

# Function that prints a message and saves it to a specified file
def print_and_save(file_path, text):
    print(text)
    with open(file_path, 'a') as file:
        file.write(text)
        file.write('\n')

# Function that logs the hyperparameters to the log file in a readable format
def log_hyperparameters(log_path, hyperparameters):
    hyperparameters_log_text = ''
    for key, value in hyperparameters.items():
        hyperparameters_log_text += f'{key}: {value}\n'
    print_and_save(log_path, hyperparameters_log_text)

# Function that logs the results after a training and validation epoch to the log file in a readable format
def log_results_train_val(log_path, epoch, train_loss, train_metrics, validation_loss, validation_metrics, start_time, end_time):
    # Write the epoch results to the log file
    epoch_duration_min = int((end_time - start_time) / 60)
    epoch_duration_sec = int((end_time - start_time) - (epoch_duration_min * 60))
    epoch_log_text = f'Epoch {epoch + 1} | Epoch Time: {epoch_duration_min}m {epoch_duration_sec}s\n'
    epoch_log_text += f'\tTrain Loss: {train_loss:.4f} - Jaccard: {train_metrics[0]:.4f} - Dice (F1): {train_metrics[1]:.4f} - Recall: {train_metrics[2]:.4f} - Precision: {train_metrics[3]:.4f}\n'
    epoch_log_text += f'\tValidation Loss: {validation_loss:.4f} - Jaccard: {validation_metrics[0]:.4f} - Dice (F1): {validation_metrics[1]:.4f} - Recall: {validation_metrics[2]:.4f} - Precision: {validation_metrics[3]:.4f}\n'
    print_and_save(log_path, epoch_log_text)

# Function that logs the test results to the log file in a readable format
def log_results_test(log_path, test_loss, test_metrics, test_mode=False):
    if test_mode:
        test_log_text = f'Test Loss: {test_loss:.4f} - Jaccard: {test_metrics[0]*100.0:.2f} - Dice (F1): {test_metrics[1]*100.0:.2f} - Recall: {test_metrics[2]*100.0:.2f} - Precision: {test_metrics[3]*100.0:.2f} - HD95: {test_metrics[4]:.2f}\n'
        #test_log_text = f'Test Loss: {test_loss:.4f} - Jaccard: {test_metrics[0]:.4f} - Dice (F1): {test_metrics[1]:.4f} - Recall: {test_metrics[2]:.4f} - Precision: {test_metrics[3]:.4f} - HD95: {test_metrics[4]:.4f}\n'
    else:
        test_log_text = f'Test Loss: {test_loss:.4f} - Jaccard: {test_metrics[0]:.4f} - Dice (F1): {test_metrics[1]:.4f} - Recall: {test_metrics[2]:.4f} - Precision: {test_metrics[3]:.4f}\n'
    print_and_save(log_path, test_log_text)


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


''' ================================== OPTIMIZER FROM TRAINABLE PARAMETERS ================================== '''

# Function that creates an optimizer for the trainable parameters
def create_optimizer(model, learning_rate):
    trainable_parameters = [param for param in model.parameters() if param.requires_grad]
    return torch.optim.Adam(trainable_parameters, lr=learning_rate)


''' ===================================== LOAD DATASET-SPECIFIC MODELS  ===================================== '''

# Function that loads dataset specific models from specified checkpoints and returns them
def load_dataset_specific_models(checkpoint_paths_by_dataset_id, models_by_dataset_id, device, load_into_submodule=None):
    for dataset_id, checkpoint_path in checkpoint_paths_by_dataset_id.items():
        state_dict = torch.load(checkpoint_path, map_location=device)

        if load_into_submodule is None:
            models_by_dataset_id[dataset_id].load_state_dict(state_dict)
        else:
            getattr(models_by_dataset_id[dataset_id], load_into_submodule).load_state_dict(state_dict)

        models_by_dataset_id[dataset_id] = freeze_model_parameters(models_by_dataset_id[dataset_id])
        
    return models_by_dataset_id


''' ============================== DETERMINE WEIGHTS FOR DATASET-SPECIFIC MODELS ============================== '''

# Determine the weights for each dataset specific model based on the specified mode and the dataset ids of the samples in the batch
def compute_dataset_specific_weights(dataset_ids, weighting_mode, num_dataset_specific_models):
    # one hot encoding = > 1 for the dataset specific model corresponding to the dataset and 0 for the others
    if weighting_mode == 0: 
        return F.one_hot(dataset_ids, num_classes=num_dataset_specific_models).float()
    # uniform weights = > 1/num_dataset_specific_models for all dataset specific models
    elif weighting_mode == 1: 
        return torch.full((dataset_ids.shape[0], num_dataset_specific_models), 1.0 / num_dataset_specific_models, device=dataset_ids.device, dtype=torch.float32)
    # biased weights = > 0.5 for the dataset specific model corresponding to the dataset and 0.25 for the others
    elif weighting_mode == 2: 
        weights = torch.full((dataset_ids.shape[0], num_dataset_specific_models), 0.25, device=dataset_ids.device, dtype=torch.float32)
        return weights.scatter_(1, dataset_ids.view(-1, 1), 0.5)