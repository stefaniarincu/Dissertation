import os
import time
import torch
from torch.utils.data import DataLoader
import albumentations as A
from utils import seed_all, create_log_file, print_and_save, log_hyperparameters, log_results_train_val, log_results_test
from data import load_split_data, shuffle_data, SegmentationDataset
from metrics import DiceBCELoss, update_metrics, compute_final_results
from models_tresunet import TResUnet
from model_unet import UNet

SEED = 42
DEVICE = torch.device('cuda')

# Constant for hyperparameters (moved here for claity and easy modification)
HYPERPARAMETERS = {
    'image_size': (256, 256),
    'batch_size': 16,
    'num_epochs': 300,
    'init_learning_rate': 0.0001,
    'scheduler_patience': 5,
    'early_stopping_patience': 20
}

# Constant for the root path for all necessary files
ROOT_PATH = '/root/Disertation'

# Constants for dataset name and path
DATASET_NAME = 'isles' # 'bmshare', 'brats'
DATASET_PATH = f'{ROOT_PATH}/datasets/{DATASET_NAME}'

# Constant for model checkpoint path and log path
MODELS_AND_LOG_ROOT_PATH = f'{ROOT_PATH}/files/dataset_specific/unet/{DATASET_NAME}'
os.makedirs(MODELS_AND_LOG_ROOT_PATH, exist_ok=True)
CHECKPOINT_PATH = f'{MODELS_AND_LOG_ROOT_PATH}/dataset_specific_model_{DATASET_NAME}.pth'
TRAIN_LOG_PATH = f'{MODELS_AND_LOG_ROOT_PATH}/train_log_{DATASET_NAME}.txt'
TEST_LOG_PATH = f'{MODELS_AND_LOG_ROOT_PATH}/test_log_{DATASET_NAME}.txt'


def train_step(model, dataloader, optimizer, criterion, device):
    model.train()
    
    epoch_loss = 0.0
    results = {'jaccard': 0.0, 'dice': 0.0, 'recall': 0.0, 'precision': 0.0}

    for batched_images, batched_masks in dataloader:
        batched_images = batched_images.to(device, dtype=torch.float32, non_blocking=True)
        batched_masks = batched_masks.to(device, dtype=torch.float32, non_blocking=True)

        optimizer.zero_grad()

        y_pred = model(batched_images)
        loss = criterion(y_pred, batched_masks)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * batched_images.size(0)

        for yt, yp in zip(batched_masks, y_pred):
            update_metrics(results, yt, yp)

    return compute_final_results(epoch_loss, results, len(dataloader.dataset))


def evaluate_step(model, dataloader, criterion, device):
    model.eval()

    epoch_loss = 0.0
    results = {'jaccard': 0.0, 'dice': 0.0, 'recall': 0.0, 'precision': 0.0}

    with torch.inference_mode():
        for batched_images, batched_masks in dataloader:
            batched_images = batched_images.to(device, dtype=torch.float32, non_blocking=True)
            batched_masks = batched_masks.to(device, dtype=torch.float32, non_blocking=True)

            y_pred = model(batched_images)
            loss = criterion(y_pred, batched_masks)

            epoch_loss += loss.item() * batched_images.size(0)

            for yt, yp in zip(batched_masks, y_pred):
                update_metrics(results, yt, yp)

    return compute_final_results(epoch_loss, results, len(dataloader.dataset))


if __name__ == '__main__':
    seed_all(SEED)
    create_log_file(TRAIN_LOG_PATH)
    log_hyperparameters(TRAIN_LOG_PATH, HYPERPARAMETERS)

    # Load the images and masks file names for training and validation
    train_images_paths, train_masks_paths = load_split_data(DATASET_PATH, 'train.txt')
    validation_images_paths, validation_masks_paths = load_split_data(DATASET_PATH, 'val.txt')
    train_images_paths, train_masks_paths = shuffle_data((train_images_paths, train_masks_paths), SEED)
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
    train_dataset = SegmentationDataset(train_images_paths, train_masks_paths, HYPERPARAMETERS['image_size'], transform=augmentation)
    validation_dataset = SegmentationDataset(validation_images_paths, validation_masks_paths, HYPERPARAMETERS['image_size'], transform=None)
    
    # Create dataloaders for training and validation datasets
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=HYPERPARAMETERS['batch_size'], shuffle=True, num_workers=2, pin_memory=True, persistent_workers=True)
    validation_dataloader = DataLoader(dataset=validation_dataset, batch_size=HYPERPARAMETERS['batch_size'], shuffle=False, num_workers=2, pin_memory=True, persistent_workers=True)

    # Create model, optimizer, scheduler and criterion
    #model = TResUnet().to(DEVICE)
    model = UNet(3, 1, True).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=HYPERPARAMETERS['init_learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=HYPERPARAMETERS['scheduler_patience'])
    criterion = DiceBCELoss()

    # Initialize variables for tracking the best validation metric and early stopping
    best_validation_metric = -1.0
    num_epochs_no_improvement = 0

    for epoch in range(HYPERPARAMETERS['num_epochs']):
        start_time = time.time()

        # Train and evaluate for one epoch
        train_loss, train_metrics = train_step(model, train_dataloader, optimizer, criterion, DEVICE)
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

    # Create the test log file
    create_log_file(TEST_LOG_PATH)

    # Load test data
    test_images_paths, test_masks_paths = load_split_data(DATASET_PATH, 'test.txt')
    dataset_log_text = f'Test set size: {len(test_images_paths)}\n'
    print_and_save(TEST_LOG_PATH, dataset_log_text)
    
    # Create dataset and dataloader for the test set of the current dataset
    test_dataset = SegmentationDataset(test_images_paths, test_masks_paths, HYPERPARAMETERS['image_size'], transform=None)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=HYPERPARAMETERS['batch_size'], shuffle=False, num_workers=0, pin_memory=True)
    
    # Load best model and check its performance
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    test_loss, test_metrics = evaluate_step(model, test_dataloader, criterion, DEVICE)
    log_results_test(TEST_LOG_PATH, test_loss, test_metrics)