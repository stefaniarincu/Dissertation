import os
import time
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import albumentations as A
from torch.amp import autocast, GradScaler
from utils import  seed_all,create_log_file, print_and_save, log_hyperparameters, load_dataset_specific_models, freeze_model_parameters, log_results_train_val, log_results_test
from data import load_split_data, shuffle_data, SegmentationDataset
from metrics import DiceBCELoss, update_metrics, compute_final_results
from models_tresunet import TResUnet, TResUnetFusedModel
#from fused_new_weighting import TResUnetFusedModel

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
#IDS_TO_DATASETS = {0: 'isles', 1: 'bmshare', 2: 'brats', 3: 'brats_ped'}
IDS_TO_DATASETS = {0: 'isles', 1: 'bmshare', 2: 'brats'}

# Constant for the root path for all necessary files
ROOT_PATH = '/root/Disertation'

# Constants for dataset name and path
DATASET_NAME = 'isles' # 'isles', 'bmshare', 'brats', 'brats_ped'
DATASET_PATH = f'{ROOT_PATH}/datasets/{DATASET_NAME}'

# Constants for model checkpoint path and log paths for the model trained on a single dataset using knowledge distillation
CHECKPOINT_PATH = f'{ROOT_PATH}/files/final_experiments/knowledge_distillation/fused_not_weighted_no_cross_attention_10K_samples_3ds_new_dropout/3_features/{DATASET_NAME}/distilled_model_{DATASET_NAME}.pth'


# Validation monitors segmentation performance only, without feature alignment loss
def evaluate_with_tta(student_model, dataloader, dice_bce_criterion, device):
    student_model.eval()

    epoch_loss = 0.0
    results = {'jaccard': 0.0, 'dice': 0.0, 'recall': 0.0, 'precision': 0.0}

    with autocast('cuda'), torch.inference_mode():
        for batched_images, batched_masks in dataloader:
            batched_images = batched_images.to(device, dtype=torch.float32, non_blocking=True)
            batched_masks = batched_masks.to(device, dtype=torch.float32, non_blocking=True)

            y_pred_original = student_model(batched_images)

            images_flipped_h = torch.flip(batched_images, dims=[3])
            y_pred_flipped_h = student_model(images_flipped_h)
            y_pred_flipped_h = torch.flip(y_pred_flipped_h, dims=[3])

            images_flipped_v = torch.flip(batched_images, dims=[2])
            y_pred_flipped_v = student_model(images_flipped_v)
            y_pred_flipped_v = torch.flip(y_pred_flipped_v, dims=[2])

            '''images_flipped_hv = torch.flip(batched_images, dims=[2, 3])
            y_pred_flipped_hv = student_model(images_flipped_hv)
            y_pred_flipped_hv = torch.flip(y_pred_flipped_hv, dims=[2, 3])'''

            #y_pred = (y_pred_original + y_pred_flipped_h + y_pred_flipped_v + y_pred_flipped_hv) / 4.0
            y_pred = (y_pred_original + y_pred_flipped_h + y_pred_flipped_v) / 3.0

            segmentation_loss = dice_bce_criterion(y_pred, batched_masks)
            epoch_loss += segmentation_loss.item() * batched_images.size(0)

            for yt, yp in zip(batched_masks, y_pred):
                update_metrics(results, yt, yp)

    return compute_final_results(epoch_loss, results, len(dataloader.dataset))


if __name__ == '__main__':
    seed_all(SEED)

    # Create model, optimizer, scheduler, and criterion
    student_model = TResUnet().to(DEVICE)
    student_model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    dice_bce_criterion = DiceBCELoss()

    # Load the images and masks file names for the test split
    test_images_paths, test_masks_paths = load_split_data(DATASET_PATH, 'test.txt')

    # Create dataset and dataloader for the test set of the current dataset
    test_dataset = SegmentationDataset(test_images_paths, test_masks_paths, HYPERPARAMETERS['image_size'], transform=None)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=HYPERPARAMETERS['batch_size'], shuffle=False, num_workers=0, pin_memory=True)

    # Test the model
    test_loss, test_metrics = evaluate_with_tta(student_model, test_dataloader, dice_bce_criterion, DEVICE)
    print(f'Test Loss: {test_loss:.4f} - Jaccard: {test_metrics[0]:.4f} - Dice (F1): {test_metrics[1]:.4f} - Recall: {test_metrics[2]:.4f} - Precision: {test_metrics[3]:.4f}\n')