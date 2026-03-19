import os
import time
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import albumentations as A
from utils import  seed_all,create_log_file, print_and_save, log_hyperparameters, log_results_test, log_results_train_val, save_resume_checkpoint, load_resume_checkpoint, load_dataset_specific_models, freeze_model_parameters
from data import load_split_data, shuffle_data, SegmentationDataset
from metrics import DiceBCELoss, compute_final_results, update_metrics
from models import TResUnet, TResUnetFusedModel

# Set a fixed seed value
SEED = 42
# Set the device to cuda
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

# Dictionary that maps dataset names to an id
IDS_TO_DATASETS = {0: 'isles', 1: 'bmshare', 2: 'brats'}

# Constant for the root path for all necessary files
ROOT_PATH = '/root/Disertation'

# Constants for dataset name and path
DATASET_NAME = 'brats' # 'bmshare', 'brats'
DATASET_PATH = f'{ROOT_PATH}/datasets/{DATASET_NAME}'

# Constant for dataset specific models checkpoint paths and a mapping from dataset names to paths
DATASET_SPECIFIC_MODELS_ROOT_PATH = f'{ROOT_PATH}/files/dataset_specific'
DATASET_SPECIFIC_MODELS_CHECKPOINTS = {dataset_id: os.path.join(DATASET_SPECIFIC_MODELS_ROOT_PATH, dataset_name, f'dataset_specific_model_{dataset_name}.pth') for dataset_id, dataset_name in IDS_TO_DATASETS.items()}

# Constant for fused model checkpoint path
FUSED_MODEL_CHECKPOINT_PATH = f'{ROOT_PATH}/files/fused_dataset_specific/not_weighted/fused_dataset_specific_model.pth'

# Constants for model checkpoint path and log paths for the model trained on a single dataset using knowledge distillation
MODELS_AND_LOG_ROOT_PATH = f'{ROOT_PATH}/files/kd/{DATASET_NAME}'
os.makedirs(MODELS_AND_LOG_ROOT_PATH, exist_ok=True)
CHECKPOINT_PATH = f'{MODELS_AND_LOG_ROOT_PATH}/distilled_model_{DATASET_NAME}.pth'
TRAIN_LOG_PATH = f'{MODELS_AND_LOG_ROOT_PATH}/train_log_{DATASET_NAME}.txt'
TEST_LOG_PATH = f'{MODELS_AND_LOG_ROOT_PATH}/test_log_{DATASET_NAME}.txt'
# Constant for resume checkpoint path if the training stops for whatever reason
RESUME_CHECKPOINT_PATH = f'{MODELS_AND_LOG_ROOT_PATH}/distilled_model_last_resume.pth'

# Feature Aligner: Projects generic model features into the dimensions of fused dataset-specific models features
'''class FeatureAligner(nn.Module):
    def __init__(self, student_dims, teacher_dims):
        super().__init__()
        self.projections = nn.ModuleList([
            nn.Conv2d(teacher_dim, student_dim, kernel_size=1) for student_dim, teacher_dim in zip(student_dims, teacher_dims)
        ])

    def forward(self, teacher_features):
        return [proj(teacher_feature) for proj, teacher_feature in zip(self.projections, teacher_features)]'''

# Feature alignment loss
def compute_feature_alignment_loss(student_features, teacher_features):
    '''for student_feature in student_features:
        print(student_feature.shape)
    for teacher_feature in teacher_features:
        print(teacher_feature.shape)'''

    '''aligners = [
        nn.Conv2d(teacher_features[i].size(1), student_features[i].size(1), kernel_size=1, stride=1, padding=0).to(device)
        for i in range(len(student_features))
    ]
    aligned_teacher_features = [aligners[i](teacher_features[i]) for i in range(len(teacher_features))]'''

    return sum(F.mse_loss(student_feature, teacher_feature) for student_feature, teacher_feature in zip(student_features, teacher_features)) / len(teacher_features)


def train_step(model, dataloader, optimizer, criterion, teacher_model, device):
    model.train()
    teacher_model.eval()
    
    epoch_loss = 0.0
    results = {'jaccard': 0.0, 'dice': 0.0, 'recall': 0.0, 'precision': 0.0}

    for batched_images, batched_masks in dataloader:
        batched_images = batched_images.to(device, dtype=torch.float32, non_blocking=True)
        batched_masks = batched_masks.to(device, dtype=torch.float32, non_blocking=True)

        optimizer.zero_grad()

        # Pass the batch through the teacher model
        with torch.no_grad():
            _, teacher_features = teacher_model(batched_images, weighting_mode=None, dataset_ids=None, return_features=True)
        
        y_pred, student_features = model(batched_images, return_features=True)
        dice_bce_loss = criterion(y_pred, batched_masks)
        feature_alignment_loss = compute_feature_alignment_loss(student_features, teacher_features)

        # Combine the segmentation loss and the feature alignment loss, perform backpropagation and update the model parameters
        total_loss = dice_bce_loss + feature_alignment_loss
        total_loss.backward()
        optimizer.step()

        epoch_loss += total_loss.item() * batched_images.size(0)

        # Calculate metrics
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
            dice_bce_loss = criterion(y_pred, batched_masks)

            epoch_loss += dice_bce_loss.item() * batched_images.size(0)

            # Compute metrics
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
    dataset_log_text = f'Dataset Size:\nTrain: {len(train_images_paths)}\nValidation: {len(validation_images_paths)}\n'
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
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=HYPERPARAMETERS['batch_size'], num_workers=2, pin_memory=True, persistent_workers=True)
    validation_dataloader = DataLoader(dataset=validation_dataset, batch_size=HYPERPARAMETERS['batch_size'], num_workers=2, pin_memory=True, persistent_workers=True)

    # Load dataset specific models checkpoints for each dataset
    dataset_specific_models = {dataset_id: TResUnet().to(DEVICE) for dataset_id in IDS_TO_DATASETS.keys()}
    dataset_specific_models = load_dataset_specific_models(DATASET_SPECIFIC_MODELS_CHECKPOINTS, dataset_specific_models, DEVICE)
    # Load the fused model checkpoint and create the teacher model for knowledge distillation    
    teacher_model = TResUnetFusedModel(dataset_specific_models).to(DEVICE)
    teacher_model.load_state_dict(torch.load(FUSED_MODEL_CHECKPOINT_PATH, map_location=DEVICE), strict=False)
    '''incompatible = teacher_model.load_state_dict( torch.load(FUSED_MODEL_CHECKPOINT_PATH, map_location=DEVICE), strict=False )
    print(incompatible.missing_keys)
    print(incompatible.unexpected_keys)'''
    teacher_model = freeze_model_parameters(teacher_model)

    # Feature aligner
    #aligner = FeatureAligner([192, 768, 1536], [192, 768, 1536]).to(DEVICE)

    # Create model, optimizer, scheduler, and criterion
    model = TResUnet().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=HYPERPARAMETERS['init_learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=HYPERPARAMETERS['scheduler_patience'])
    criterion = DiceBCELoss()

    # Initialize variables for tracking the starting epoch, best validation metric and early stopping
    start_epoch = 0
    best_validation_metric = -1.0
    num_epochs_no_improvement = 0

    # If resume checkpoint exists, load it
    if os.path.exists(RESUME_CHECKPOINT_PATH):
        start_epoch, best_validation_metric, num_epochs_no_improvement = load_resume_checkpoint(model, optimizer, scheduler, RESUME_CHECKPOINT_PATH, DEVICE)

    for epoch in range(start_epoch, HYPERPARAMETERS['num_epochs']):
        start_time = time.time()

        # Train and evaluate for one epoch
        train_loss, train_metrics = train_step(model, train_dataloader, optimizer, criterion, teacher_model, DEVICE)
        validation_loss, validation_metrics = evaluate_step(model, validation_dataloader, criterion, DEVICE)
        scheduler.step(validation_loss)

        # If the validation Dice (F1) score improved, save the model checkpoint and reset the early stopping counter
        if validation_metrics[1] > best_validation_metric:
            best_validation_metric = validation_metrics[1]
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            num_epochs_no_improvement = 0

            data_str = f'Valid F1 improved from {best_validation_metric:2.4f} to {validation_metrics[1]:2.4f}. Saving checkpoint: {CHECKPOINT_PATH}'
            print_and_save(TRAIN_LOG_PATH, data_str)
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

    # Load the images and masks file names for the test split
    test_images_paths, test_masks_paths = load_split_data(DATASET_PATH, 'test.txt')
    dataset_log_text = f'Test set size: {len(test_images_paths)}\n'
    print_and_save(TEST_LOG_PATH, dataset_log_text)

    # Create dataset and dataloader for the test set of the current dataset
    test_dataset = SegmentationDataset(test_images_paths, test_masks_paths, HYPERPARAMETERS['image_size'], transform=None)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=HYPERPARAMETERS['batch_size'], shuffle=False, num_workers=0, pin_memory=True)

    # Test the model
    test_loss, test_metrics = evaluate_step(model, test_dataloader, criterion, DEVICE)
    log_results_test(TEST_LOG_PATH, test_loss, test_metrics)