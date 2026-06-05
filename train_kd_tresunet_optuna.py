import os
import time
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import albumentations as A
from torch.amp import autocast, GradScaler
import optuna
from utils import  seed_all,create_log_file, print_and_save, log_hyperparameters, load_dataset_specific_models, freeze_model_parameters, log_results_train_val, log_results_test
from data import load_split_data, shuffle_data, SegmentationDataset
from metrics import DiceBCELoss, update_metrics, compute_final_results
from models_tresunet import TResUnet, TResUnetFusedModel
#from fused_new_weighting import TResUnetFusedModel

import pandas as pd

SEED = 42
DEVICE = torch.device('cuda')

grad_scaler = GradScaler('cuda')

# Constant for hyperparameters (moved here for clarity and easy modification)
HYPERPARAMETERS = {
    'image_size': (256, 256),
    'batch_size': 16,
    'num_epochs': 100,
    'init_learning_rate': 0.0001,
    'scheduler_patience': 5,
    'early_stopping_patience': 20,
    'alpha': 0.5,
    'gamma': 0.3,
    'delta': 0.1,
    'initial_temperature': 2.0,
    'initial_contrastive_weight': 0.1,
    'max_contrastive_weight': 0.5,
    'weight_increment': 0.01
}


# Dictionary that maps dataset names to an id
IDS_TO_DATASETS = {0: 'isles', 1: 'bmshare', 2: 'brats'}

# Constant for the root path for all necessary files
ROOT_PATH = '/root/Disertation'
EXPERIMENTS_ROOT_PATH = f'{ROOT_PATH}/files/final_experiments'

# Constants for dataset name and path
DATASET_NAME = 'isles' # 'isles', 'bmshare', 'brats', 'brats_ped'
DATASET_PATH = f'{ROOT_PATH}/datasets/{DATASET_NAME}'

# Constant for dataset specific models checkpoint paths and a mapping from dataset names to paths
DATASET_SPECIFIC_MODELS_ROOT_PATH = f'{EXPERIMENTS_ROOT_PATH}/dataset_specific/fract'
DATASET_SPECIFIC_MODELS_CHECKPOINTS = {dataset_id: os.path.join(DATASET_SPECIFIC_MODELS_ROOT_PATH, dataset_name, f'dataset_specific_model_{dataset_name}.pth') for dataset_id, dataset_name in IDS_TO_DATASETS.items()}

# Constant for fused model checkpoint path
FUSED_MODEL_CHECKPOINT_PATH = f'{EXPERIMENTS_ROOT_PATH}/fused/not_weighted_no_cross_attention_10K_samples_3ds_new_dropout/fused_model.pth'

# Constants for model checkpoint path and log paths for the model trained on a single dataset using knowledge distillation
LOG_ROOT_PATH = f'{EXPERIMENTS_ROOT_PATH}/knowledge_distillation/from_fused_not_weighted_no_cross_attention_10K_samples_3ds_new_dropout/kd_optuna2'
os.makedirs(LOG_ROOT_PATH, exist_ok=True)

def suggest_kd_params(trial):
    params = {
        'alpha': trial.suggest_float('alpha', 0.4, 1.0, step=0.1),
        'gamma': trial.suggest_float('gamma', 0.1, 1.0, step=0.1),
        'delta': trial.suggest_float('delta', 0.05, 1.0, step=0.05),

        'initial_temperature': trial.suggest_float('initial_temperature', 1.5, 4.0, step=0.1),
        'min_temperature': trial.suggest_float('min_temperature', 0.5, 1.5, step=0.1),
        'temperature_decay': trial.suggest_float('temperature_decay', 0.85, 0.98, step=0.01),
        'temperature_decay_step': trial.suggest_int('temperature_decay_step', 3, 10, step=1),

        'initial_contrastive_weight': trial.suggest_float('initial_contrastive_weight', 0.03, 0.2, step=0.01),
        'max_contrastive_weight': trial.suggest_float('max_contrastive_weight', 0.3, 1.0, step=0.05),
        'weight_increment': trial.suggest_float('weight_increment', 0.003, 0.02, step=0.001),

        'warmup_epochs': trial.suggest_int('warmup_epochs', 3, 8, step=1),
        'ramp_epochs': trial.suggest_int('ramp_epochs', 8, 20, step=1)
    }
    '''params = {
        'alpha': trial.suggest_float('alpha', 0.4, 1.0, step=0.05),
        'gamma': trial.suggest_float('gamma', 0.1, 1.0, step=0.05),
        'delta': trial.suggest_float('delta', 0.03, 1.0, step=0.01),

        'initial_temperature': trial.suggest_float('initial_temperature', 1.5, 4.0, step=0.05),
        'min_temperature': trial.suggest_float('min_temperature', 0.5, 1.5, step=0.05),
        'temperature_decay': trial.suggest_float('temperature_decay', 0.85, 0.98, step=0.01),
        'temperature_decay_step': trial.suggest_int('temperature_decay_step', 3, 10, step=1),

        'initial_contrastive_weight': trial.suggest_float('initial_contrastive_weight', 0.03, 0.2, step=0.01),
        'max_contrastive_weight': trial.suggest_float('max_contrastive_weight', 0.3, 1.0, step=0.01),
        'weight_increment': trial.suggest_float('weight_increment', 0.003, 0.02, step=0.001),

        'warmup_epochs': trial.suggest_int('warmup_epochs', 3, 8, step=1),
        'ramp_epochs': trial.suggest_int('ramp_epochs', 8, 20, step=1),

        'init_learning_rate': trial.suggest_categorical('init_learning_rate', [1e-4, 9e-5, 8e-5, 7e-5, 6e-5, 5e-5])
    }'''
    
    return params

# Function for dynamic curriculum for scheduling KD losses
def dynamic_curriculum(epoch, warmup_epochs=5, ramp_epochs=10):
    if epoch < warmup_epochs:
        return 0.0  # No KD in warmup
    elif epoch < warmup_epochs + ramp_epochs:
        return (epoch - warmup_epochs) / ramp_epochs  # Gradually ramp up KD
    else:
        return 1.0  # Full KD after ramp-up

def get_kd_scheduler_values(epoch, params):
    curriculum_factor = dynamic_curriculum(epoch, warmup_epochs=params['warmup_epochs'], ramp_epochs=params['ramp_epochs'])
    temperature = max(params['min_temperature'], params['initial_temperature'] * (params['temperature_decay'] ** (epoch // params['temperature_decay_step'])))
    contrastive_weight = min(params['max_contrastive_weight'], params['initial_contrastive_weight'] + epoch * params['weight_increment'])
    return curriculum_factor, temperature, contrastive_weight

# Function that flattens features for contrastive loss
def flatten_features(features):
    flat_features = [f.view(f.size(0), -1) for f in features]
    return torch.cat(flat_features, dim=1)

# Function that computes the contrastive loss by calculating the KL divergence between the similarity matrices
def contrastive_loss(student_features, teacher_features, temperature=0.5):
    student_features_concat = F.normalize(flatten_features(student_features), p=2, dim=1)
    teacher_features_concat = F.normalize(flatten_features(teacher_features), p=2, dim=1)

    similarity_matrix_student = torch.matmul(student_features_concat, student_features_concat.t())
    similarity_matrix_teacher = torch.matmul(teacher_features_concat, teacher_features_concat.t())

    similarity_student = F.softmax(similarity_matrix_student / temperature, dim=1)
    similarity_teacher = F.softmax(similarity_matrix_teacher / temperature, dim=1)
    return F.kl_div(similarity_student.log(), similarity_teacher, reduction='batchmean')
    #log_probs_student = F.log_softmax(similarity_matrix_student / temperature, dim=1)
    #probs_teacher = F.softmax(similarity_matrix_teacher / temperature, dim=1)
    #return F.kl_div(log_probs_student, probs_teacher, reduction='batchmean') * (temperature ** 2)

# Function that computes the feature alignment loss by calculating the MSE between the student and teacher features
def compute_feature_alignment_loss(student_features, teacher_features):
    return sum(F.mse_loss(student_feature, teacher_feature) for student_feature, teacher_feature in zip(student_features, teacher_features)) / len(student_features)

# Function that computes the cosine similarity between the student and teacher features
def cosine_similarity_loss(student_features, teacher_features):
    return sum(1 - F.cosine_similarity(student_feature, teacher_feature, dim=1).mean() for student_feature, teacher_feature in zip(student_features, teacher_features)) / len(student_features)

# Training uses both segmentation loss and feature alignment loss (mse, cosine similarity and contrastive loss), with dynamic curriculum scheduling for KD
def train_step(teacher_model, student_model, dataloader, optimizer, dice_bce_criterion, device, 
               epoch, alpha, gamma, delta, contrastive_weight, temperature, curriculum_factor):
    teacher_model.eval()
    student_model.train()
    
    epoch_loss = 0.0
    results = {'jaccard': 0.0, 'dice': 0.0, 'recall': 0.0, 'precision': 0.0}

    for batched_images, batched_masks in dataloader:
        batched_images = batched_images.to(device, dtype=torch.float32, non_blocking=True)
        batched_masks = batched_masks.to(device, dtype=torch.float32, non_blocking=True)

        optimizer.zero_grad()

        # Pass the batch through the teacher model
        with autocast('cuda'), torch.no_grad():
            teacher_output, teacher_features = teacher_model(batched_images, dataset_ids=None, weighting_mode=None, return_features=True)
            #teacher_probs = torch.sigmoid(teacher_output).detach()
            teacher_features = [teacher_feature.detach() for teacher_feature in teacher_features]
            teacher_features = teacher_features[:3]
            #print(teacher_features[0].shape, teacher_features[1].shape, teacher_features[2].shape)

        with autocast('cuda'):
            student_output, student_features = student_model(batched_images, return_features=True)
            student_features = student_features[:3]
            #print(student_features[0].shape, student_features[1].shape, student_features[2].shape)

            segmentation_loss = dice_bce_criterion(student_output, batched_masks)
            kd_contrastive_loss = contrastive_loss(student_features, teacher_features, temperature=temperature)
            feature_alignment_loss = compute_feature_alignment_loss(student_features, teacher_features)
            similarity_loss = cosine_similarity_loss(student_features, teacher_features)

            # Combine the segmentation loss and the feature alignment loss, perform backpropagation and update the model parameters
            total_loss = (alpha * segmentation_loss +
                        curriculum_factor * contrastive_weight * kd_contrastive_loss +
                        curriculum_factor * gamma * feature_alignment_loss +
                        curriculum_factor * delta * similarity_loss)
            #print(f'Segmentation: {segmentation_loss.item():.4f}, Contrastive: {kd_contrastive_loss.item():.4f}, Feature Alignment: {feature_alignment_loss.item():.4f}, Similarity: {similarity_loss.item():.4f}, Total: {total_loss.item():.4f}')
            
        grad_scaler.scale(total_loss).backward()
        grad_scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)

        grad_scaler.step(optimizer)
        grad_scaler.update()

        epoch_loss += total_loss.item() * batched_images.size(0)

        for yt, yp in zip(batched_masks, student_output):
            update_metrics(results, yt, yp)

    return compute_final_results(epoch_loss, results, len(dataloader.dataset))


# Validation monitors segmentation performance only, without feature alignment loss
def evaluate_step(student_model, dataloader, dice_bce_criterion, device):
    student_model.eval()

    epoch_loss = 0.0
    results = {'jaccard': 0.0, 'dice': 0.0, 'recall': 0.0, 'precision': 0.0}

    with autocast('cuda'), torch.inference_mode():
        for batched_images, batched_masks in dataloader:
            batched_images = batched_images.to(device, dtype=torch.float32, non_blocking=True)
            batched_masks = batched_masks.to(device, dtype=torch.float32, non_blocking=True)

            y_pred = student_model(batched_images)
            segmentation_loss = dice_bce_criterion(y_pred, batched_masks)

            epoch_loss += segmentation_loss.item() * batched_images.size(0)

            for yt, yp in zip(batched_masks, y_pred):
                update_metrics(results, yt, yp)

    return compute_final_results(epoch_loss, results, len(dataloader.dataset))

def objective(trial):
    params = suggest_kd_params(trial)

    trial_log_path = os.path.join(LOG_ROOT_PATH, f'train_log_trial_{trial.number}.txt')
    create_log_file(trial_log_path)
    print_and_save(trial_log_path, f'Trial {trial.number}\nHyperparameters: {params}\n')

    seed_all(SEED)

    # Load the images and masks file names for training and validation
    train_images_paths, train_masks_paths = load_split_data(DATASET_PATH, 'train.txt')
    validation_images_paths, validation_masks_paths = load_split_data(DATASET_PATH, 'val.txt')
    train_images_paths, train_masks_paths = shuffle_data((train_images_paths, train_masks_paths), SEED)

    # Define data augmentation transforms using albumentations
    augmentation = A.Compose([
        A.Rotate(limit=35, p=0.3),
        A.HorizontalFlip(p=0.3),
        A.VerticalFlip(p=0.3),
    ])

    # Create datasets for training and validation
    train_dataset = SegmentationDataset(train_images_paths, train_masks_paths, HYPERPARAMETERS['image_size'], transform=augmentation)
    validation_dataset = SegmentationDataset(validation_images_paths, validation_masks_paths, HYPERPARAMETERS['image_size'], transform=None)

    # Create dataloaders for training and validation datasets
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=HYPERPARAMETERS['batch_size'], shuffle=True, num_workers=2, pin_memory=True, persistent_workers=True)
    validation_dataloader = DataLoader(dataset=validation_dataset, batch_size=HYPERPARAMETERS['batch_size'], shuffle=False, num_workers=2, pin_memory=True, persistent_workers=True)

    # Create model, optimizer, scheduler, and criterion
    student_model = TResUnet().to(DEVICE)
    optimizer = torch.optim.Adam(student_model.parameters(), lr=HYPERPARAMETERS['init_learning_rate'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=HYPERPARAMETERS['num_epochs'], eta_min=1e-8)
    dice_bce_criterion = DiceBCELoss()

    # Initialize variables for tracking the best validation metric and early stopping
    best_validation_metric = -1.0
    num_epochs_no_improvement = 0
    best_state_dict = None

    for epoch in range(HYPERPARAMETERS['num_epochs']):
        start_time = time.time()

        # Dynamic curriculum scheduling
        curriculum_factor, temperature, contrastive_weight = get_kd_scheduler_values(epoch, params)

        # Train and evaluate for one epoch
        train_loss, train_metrics = train_step(teacher_model, student_model, train_dataloader, optimizer, dice_bce_criterion, DEVICE, epoch, alpha=params['alpha'], gamma=params['gamma'], delta=params['delta'], contrastive_weight=contrastive_weight, temperature=temperature, curriculum_factor=curriculum_factor)
        validation_loss, validation_metrics = evaluate_step(student_model, validation_dataloader, dice_bce_criterion, DEVICE)
        scheduler.step()

        validation_dice = validation_metrics[1]
        trial.report(validation_dice, epoch)

        if validation_dice > best_validation_metric:
            print_and_save(trial_log_path, f'Valid Dice improved from {best_validation_metric:.4f} to {validation_dice:.4f}')
            best_validation_metric = validation_dice
            num_epochs_no_improvement = 0
            best_state_dict = {k: v.detach().cpu().clone() for k, v in student_model.state_dict().items()}
        else:
            num_epochs_no_improvement += 1

        end_time = time.time()
        log_results_train_val(trial_log_path, epoch, train_loss, train_metrics, validation_loss, validation_metrics, start_time, end_time)

        if num_epochs_no_improvement == HYPERPARAMETERS['early_stopping_patience']:
            print_and_save(trial_log_path, f'Early stopping triggered after {epoch+1} epochs with no improvement.')
            break
    
    if best_state_dict is not None:
        # Load the best model state dict before testing
        student_model.load_state_dict(best_state_dict)

        test_images_paths, test_masks_paths = load_split_data(DATASET_PATH, 'test.txt')

        # Create dataset and dataloader for the test set of the current dataset
        test_dataset = SegmentationDataset(test_images_paths, test_masks_paths, HYPERPARAMETERS['image_size'], transform=None)
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=HYPERPARAMETERS['batch_size'], shuffle=False, num_workers=0, pin_memory=True)

        # Test the model
        test_loss, test_metrics = evaluate_step(student_model, test_dataloader, dice_bce_criterion, DEVICE)
        log_results_test(trial_log_path, test_loss, test_metrics)

    del student_model, optimizer, scheduler, best_state_dict
    torch.cuda.empty_cache()

    return best_validation_metric


if __name__ == '__main__':
    # Load dataset specific models checkpoints for each dataset
    dataset_specific_models = {dataset_id: TResUnet().to(DEVICE) for dataset_id in IDS_TO_DATASETS.keys()}
    dataset_specific_models = load_dataset_specific_models(DATASET_SPECIFIC_MODELS_CHECKPOINTS, dataset_specific_models, DEVICE)

    # Load the fused model checkpoint and create the teacher model for knowledge distillation
    teacher_model = TResUnetFusedModel(list(dataset_specific_models.values())).to(DEVICE)
    teacher_model.load_state_dict(torch.load(FUSED_MODEL_CHECKPOINT_PATH, map_location=DEVICE))#, strict=False)
    '''incompatible = teacher_model.load_state_dict(torch.load(FUSED_MODEL_CHECKPOINT_PATH, map_location=DEVICE))#, strict=False )
    print(incompatible.missing_keys)
    print(incompatible.unexpected_keys)'''
    teacher_model = freeze_model_parameters(teacher_model)

    optuna_sampler = optuna.samplers.TPESampler(seed=SEED, multivariate=True)
    study = optuna.create_study(study_name='kd_params', direction='maximize', sampler=optuna_sampler, storage=f'sqlite:///{LOG_ROOT_PATH}/optuna_study.db', load_if_exists=True)
    
    remaining_trials = 65 - len(study.trials)
    if remaining_trials > 0:
        print(f'Remaining trials to run: {remaining_trials}')
        study.optimize(objective, n_trials=remaining_trials)

    print(f'Best trial: {study.best_trial.number}, Value: {study.best_trial.value}')
    print(f'Best hyperparameters: {study.best_trial.params}')

    study.trials_dataframe().to_csv(os.path.join(LOG_ROOT_PATH, 'optuna_trials_summary.csv'), index=False)