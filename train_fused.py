import os
import time
import torch
from torch.utils.data import DataLoader
import albumentations as A
from utils import seed_all, create_log_file, print_and_save, log_hyperparameters, save_resume_checkpoint, load_resume_checkpoint, load_dataset_specific_models, create_optimizer, log_results_train_val, log_results_test
from data import load_split_data, load_split_data_all_datasets, shuffle_data, SegmentationDatasetWithDatasetId, BalancedBatchSampler
from metrics import DiceBCELoss, update_metrics, compute_final_results
from models_tresunet import TResUnet, TResUnetFusedModel

SEED = 42
DEVICE = torch.device('cuda')

# Constant for hyperparameters (moved here for claity and easy modification)
HYPERPARAMETERS = {
    'image_size': (256, 256),
    'batch_size': 16,
    'num_epochs': 300,
    'init_learning_rate': 0.0001,
    'scheduler_patience': 5,
    'early_stopping_patience': 20,
    # dataset specific models weighting - 0 one-hot (match own dataset expert), 1 uniform, 2 biasd towards own dataset expert and None if no weighted mode wanted
    'dsm_weighting_mode': None,
}

# Dictionary that maps dataset names to an id
IDS_TO_DATASETS = {0: 'isles', 1: 'bmshare', 2: 'brats'}

# Constant for the root path for all necessary files
ROOT_PATH = '/root/Disertation'

# Constants for paths of all datasets
DATASETS_ROOT_PATH = f'{ROOT_PATH}/datasets'
DATASETS_PATHS = {dataset_id: os.path.join(DATASETS_ROOT_PATH, dataset_name) for dataset_id, dataset_name in IDS_TO_DATASETS.items()}

# Constant for dataset specific models checkpoint paths and a mapping from dataset names to paths
DATASET_SPECIFIC_MODELS_ROOT_PATH = f'{ROOT_PATH}/files/dataset_specific'
DATASET_SPECIFIC_MODELS_CHECKPOINTS = {dataset_id: os.path.join(DATASET_SPECIFIC_MODELS_ROOT_PATH, dataset_name, f'dataset_specific_model_{dataset_name}.pth') for dataset_id, dataset_name in IDS_TO_DATASETS.items()}

# Constants for model checkpoint path and log path
MODELS_AND_LOG_ROOT_PATH = f'{ROOT_PATH}/files/fused_models/not_weighted_no_cross_attention_10K_samples'
os.makedirs(MODELS_AND_LOG_ROOT_PATH, exist_ok=True)
CHECKPOINT_PATH = f'{MODELS_AND_LOG_ROOT_PATH}/fused_model.pth'
TRAIN_LOG_PATH = f'{MODELS_AND_LOG_ROOT_PATH}/train_log_fused.txt'
TEST_LOG_PATH = f'{MODELS_AND_LOG_ROOT_PATH}/test_log_fused.txt'
# Constant for resume checkpoint path if the training stops for whatever reason
RESUME_CHECKPOINT_PATH = f'{MODELS_AND_LOG_ROOT_PATH}/fused_model_last_resume.pth'


def train_step(model, dataloader, optimizer, criterion, weighting_mode, device):
    model.train()
    
    epoch_loss = 0.0
    results = {'jaccard': 0.0, 'dice': 0.0, 'recall': 0.0, 'precision': 0.0}
    processed_samples = 0

    for batched_images, batched_masks, batched_dataset_ids in dataloader:
        batched_images = batched_images.to(device, dtype=torch.float32, non_blocking=True)
        batched_masks = batched_masks.to(device, dtype=torch.float32, non_blocking=True)
        batched_dataset_ids = batched_dataset_ids.to(device, non_blocking=True, dtype=torch.long)

        optimizer.zero_grad()

        y_pred = model(batched_images, dataset_ids=batched_dataset_ids, weighting_mode=weighting_mode)
        loss = criterion(y_pred, batched_masks)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * batched_images.size(0)
        processed_samples += batched_images.size(0)
        
        for yt, yp in zip(batched_masks, y_pred):
            update_metrics(results, yt, yp)
    
    return compute_final_results(epoch_loss, results, processed_samples)


def evaluate_step(model, dataloader, criterion, device):
    model.eval()

    epoch_loss = 0.0
    results = {'jaccard': 0.0, 'dice': 0.0, 'recall': 0.0, 'precision': 0.0}
    processed_samples = 0

    with torch.inference_mode():
        for batched_images, batched_masks, _ in dataloader:
            batched_images = batched_images.to(device, dtype=torch.float32, non_blocking=True)
            batched_masks = batched_masks.to(device, dtype=torch.float32, non_blocking=True)
            #batched_dataset_ids = batched_dataset_ids.to(device, non_blocking=True, dtype=torch.long)

            y_pred = model(batched_images)
            loss = criterion(y_pred, batched_masks)
            
            epoch_loss += loss.item() * batched_images.size(0)
            processed_samples += batched_images.size(0)

            for yt, yp in zip(batched_masks, y_pred):
                update_metrics(results, yt, yp)
    
    return compute_final_results(epoch_loss, results, processed_samples)


if __name__ == '__main__':
    seed_all(SEED)
    create_log_file(TRAIN_LOG_PATH)
    log_hyperparameters(TRAIN_LOG_PATH, HYPERPARAMETERS)

    # Load the images and masks file names for training and validation
    train_images_paths, train_masks_paths, train_dataset_ids = load_split_data_all_datasets(DATASETS_PATHS, 'train.txt')
    validation_images_paths, validation_masks_paths, validation_dataset_ids = load_split_data_all_datasets(DATASETS_PATHS, 'val.txt')
    train_images_paths, train_masks_paths, train_dataset_ids = shuffle_data((train_images_paths, train_masks_paths, train_dataset_ids), SEED)
    dataset_log_text = f'Train set size: {len(train_images_paths)}\nValidation set size: {len(validation_images_paths)}\n'
    print_and_save(TRAIN_LOG_PATH, dataset_log_text)

    # Define data augmentation transforms using albumentations
    augmentation = A.Compose([
        A.Rotate(limit=35, p=0.3),
        A.HorizontalFlip(p=0.3),
        A.VerticalFlip(p=0.3),
        A.CoarseDropout(p=0.3, num_holes_range=(1, 10), hole_height_range=(1, 32), hole_width_range=(1, 32))
    ])

    # Create datasets for training and validation
    train_dataset = SegmentationDatasetWithDatasetId(train_images_paths, train_masks_paths, train_dataset_ids, HYPERPARAMETERS['image_size'], transform=augmentation)
    validation_dataset = SegmentationDatasetWithDatasetId(validation_images_paths, validation_masks_paths, validation_dataset_ids, HYPERPARAMETERS['image_size'], transform=None)
    
    # Create balanced batch samplers for training and validation
    train_sampler = BalancedBatchSampler(train_dataset.dataset_ids, HYPERPARAMETERS['batch_size'], seed=SEED, shuffle=True, allow_incomplete_last_batch=True)
    validation_sampler = BalancedBatchSampler(validation_dataset.dataset_ids, HYPERPARAMETERS['batch_size'], seed=SEED, shuffle=False, allow_incomplete_last_batch=True)

    # Create dataloaders for training and validation datasets
    train_dataloader = DataLoader(dataset=train_dataset, batch_sampler=train_sampler, num_workers=2, pin_memory=True, persistent_workers=True)
    validation_dataloader = DataLoader(dataset=validation_dataset, batch_sampler=validation_sampler, num_workers=2, pin_memory=True, persistent_workers=True)

    # Load dataset specific models checkpoints for each dataset
    dataset_specific_models = {dataset_id: TResUnet().to(DEVICE) for dataset_id in IDS_TO_DATASETS.keys()}
    dataset_specific_models = load_dataset_specific_models(DATASET_SPECIFIC_MODELS_CHECKPOINTS, dataset_specific_models, DEVICE)

    # Create model, optimizer, scheduler, and criterion
    model = TResUnetFusedModel(dataset_specific_models).to(DEVICE)
    optimizer = create_optimizer(model, HYPERPARAMETERS['init_learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=HYPERPARAMETERS['scheduler_patience'])
    criterion = DiceBCELoss()

    # Initialize variables for the starting epoch and tracking the best validation metric and early stopping
    start_epoch = 0
    best_validation_metric = -1.0
    num_epochs_no_improvement = 0

    # If resume checkpoint exists, load it
    if os.path.exists(RESUME_CHECKPOINT_PATH):
        # Replace the variables with the values from the loaded checkpoint
        start_epoch, best_validation_metric, num_epochs_no_improvement = load_resume_checkpoint(model, optimizer, scheduler, RESUME_CHECKPOINT_PATH, DEVICE)

    for epoch in range(start_epoch, HYPERPARAMETERS['num_epochs']):
        start_time = time.time()
        train_sampler.set_epoch(epoch)

        # Train and evaluate for one epoch
        train_loss, train_metrics = train_step(model, train_dataloader, optimizer, criterion, HYPERPARAMETERS['dsm_weighting_mode'], DEVICE)
        validation_loss, validation_metrics = evaluate_step(model, validation_dataloader, criterion, DEVICE)
        scheduler.step(validation_loss)

        # If the validation Dice (F1) score improved, save the model checkpoint and reset the early stopping counter
        if validation_metrics[1] > best_validation_metric:
            data_str = f'Valid F1 improved from {best_validation_metric:2.4f} to {validation_metrics[1]:2.4f}. Saving checkpoint: {CHECKPOINT_PATH}'
            print_and_save(TRAIN_LOG_PATH, data_str)

            best_validation_metric = validation_metrics[1]
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            num_epochs_no_improvement = 0
        else:
            num_epochs_no_improvement += 1

        # Write the epoch results to the log file
        end_time = time.time()
        log_results_train_val(TRAIN_LOG_PATH, epoch, train_loss, train_metrics, validation_loss, validation_metrics, start_time, end_time)

        # If early stopping is triggered, break the training loop
        if num_epochs_no_improvement == HYPERPARAMETERS['early_stopping_patience']:
            print_and_save(TRAIN_LOG_PATH, f'Early stopping triggered after {epoch + 1} epochs.')
            break

        save_resume_checkpoint(model, epoch, optimizer, scheduler, best_validation_metric, num_epochs_no_improvement, RESUME_CHECKPOINT_PATH)

    # Create the test log file
    create_log_file(TEST_LOG_PATH)
    # Load best model and check its performance
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))

    # Iterate through all datasets and evaluate the best model on the test set of each dataset separately, logging the results
    for dataset_id, dataset_path in DATASETS_PATHS.items():
        dataset_log_text = f'{IDS_TO_DATASETS[dataset_id]} dataset'
        print_and_save(TEST_LOG_PATH, dataset_log_text)

        # Load the images and masks file names for the test split
        test_images_paths, test_masks_paths = load_split_data(dataset_path, 'test.txt')
        test_dataset_ids = [dataset_id] * len(test_images_paths)
        dataset_log_text = f'Test set size: {len(test_images_paths)}\n'
        print_and_save(TEST_LOG_PATH, dataset_log_text)

        # Create dataset and dataloader for the test set of the current dataset
        test_dataset = SegmentationDatasetWithDatasetId(test_images_paths, test_masks_paths, test_dataset_ids, HYPERPARAMETERS['image_size'], transform=None)
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=HYPERPARAMETERS['batch_size'], shuffle=False, num_workers=0, pin_memory=True)
    
        # Test the model
        test_loss, test_metrics = evaluate_step(model, test_dataloader, criterion, DEVICE)
        log_results_test(TEST_LOG_PATH, test_loss, test_metrics)