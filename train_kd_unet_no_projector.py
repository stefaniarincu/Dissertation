import os
import time
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch import nn
import albumentations as A
from utils import  seed_all,create_log_file, print_and_save, log_hyperparameters, load_dataset_specific_models, freeze_model_parameters, log_results_train_val, log_results_test
from data import load_split_data, shuffle_data, SegmentationDataset
from metrics import DiceBCELoss, update_metrics, compute_final_results
from models_tresunet import TResUnet, TResUnetFusedModel
from model_unet import UNet

SEED = 42
DEVICE = torch.device('cuda')

# Constant for hyperparameters (moved here for clarity and easy modification)
HYPERPARAMETERS = {
    'image_size': (256, 256),
    'batch_size': 16,
    'num_epochs': 100,
    'init_learning_rate': 0.0001,
    'scheduler_patience': 5,
    'early_stopping_patience': 20,
    'alpha': 0.5,
    'initial_temperature': 2,
    'initial_contrastive_weight': 0.1,
    'max_contrastive_weight': 0.5,
    'weight_increment': 0.01,
    'map_loss_weight': 0.3
}


# Dictionary that maps dataset names to an id
IDS_TO_DATASETS = {0: 'isles', 1: 'bmshare', 2: 'brats'}

# Constant for the root path for all necessary files
ROOT_PATH = '/root/Disertation'

# Constants for dataset name and path
DATASET_NAME = 'isles' # 'bmshare', 'brats'
DATASET_PATH = f'{ROOT_PATH}/datasets/{DATASET_NAME}'

# Constant for dataset specific models checkpoint paths and a mapping from dataset names to paths
DATASET_SPECIFIC_MODELS_ROOT_PATH = f'{ROOT_PATH}/files/dataset_specific/fract'
DATASET_SPECIFIC_MODELS_CHECKPOINTS = {dataset_id: os.path.join(DATASET_SPECIFIC_MODELS_ROOT_PATH, dataset_name, f'dataset_specific_model_{dataset_name}.pth') for dataset_id, dataset_name in IDS_TO_DATASETS.items()}

# Constant for fused model checkpoint path
FUSED_MODEL_CHECKPOINT_PATH = f'{ROOT_PATH}/files/fused_models/not_weighted_no_cross_attention_10K_samples_3ds_new_dropout/fused_model.pth'

# Constants for model checkpoint path and log paths for the model trained on a single dataset using knowledge distillation
MODELS_AND_LOG_ROOT_PATH = f'{ROOT_PATH}/files/kd/fused_not_weighted_no_cross_attention_10K_samples_3ds_new_dropout/unet/{DATASET_NAME}'
os.makedirs(MODELS_AND_LOG_ROOT_PATH, exist_ok=True)
CHECKPOINT_PATH = f'{MODELS_AND_LOG_ROOT_PATH}/distilled_model_{DATASET_NAME}.pth'
TRAIN_LOG_PATH = f'{MODELS_AND_LOG_ROOT_PATH}/train_log_{DATASET_NAME}.txt'
TEST_LOG_PATH = f'{MODELS_AND_LOG_ROOT_PATH}/test_log_{DATASET_NAME}.txt'


# Flatten features for contrastive loss
def flatten_features(features):
    flat_features = [f.reshape(f.size(0), -1) for f in features]
    return torch.cat(flat_features, dim=1)

# Contrastive loss
def contrastive_loss(student_features, teacher_features, temperature=0.5):
    student_features_concat = F.normalize(flatten_features(student_features), p=2, dim=1)
    teacher_features_concat = F.normalize(flatten_features(teacher_features), p=2, dim=1)

    similarity_matrix_student = torch.matmul(student_features_concat, student_features_concat.t())
    similarity_matrix_teacher = torch.matmul(teacher_features_concat, teacher_features_concat.t())

    similarity_student = F.softmax(similarity_matrix_student / temperature, dim=1)
    similarity_teacher = F.softmax(similarity_matrix_teacher / temperature, dim=1)
    return F.kl_div(similarity_student.log(), similarity_teacher, reduction='batchmean')

# Feature alignment loss
def compute_feature_alignment_loss(student_features, teacher_features, device):
    '''for student_feature in student_features:
        print(student_feature.shape)
    for teacher_feature in teacher_features:
        print(teacher_feature.shape)'''

    aligners = [
        nn.Conv2d(teacher_features[i].size(1), student_features[i].size(1), kernel_size=1, stride=1, padding=0).to(device)
        for i in range(len(student_features))
    ]
    aligned_teacher_features = [aligners[i](teacher_features[i]) for i in range(len(teacher_features))]
    return sum(F.mse_loss(student_feature, teacher_feature) for student_feature, teacher_feature in zip(student_features, aligned_teacher_features)) / len(student_features)

# Cosine similarity loss
def cosine_similarity_loss(student_features, teacher_features, device):
    aligners = [
        nn.Conv2d(teacher_features[i].size(1), student_features[i].size(1), kernel_size=1, stride=1, padding=0).to(
            device)
        for i in range(len(student_features))
    ]
    aligned_teacher_features = [aligners[i](teacher_features[i]) for i in range(len(teacher_features))]
    return sum(1 - F.cosine_similarity(student_feature, teacher_feature, dim=1).mean() for student_feature, teacher_feature in zip(student_features, aligned_teacher_features)) / len(student_features)

# Dynamic curriculum for scheduling KD losses
def dynamic_curriculum(epoch, warmup_epochs=5, ramp_epochs=10):
    if epoch < warmup_epochs:
        return 0.0  # No KD in warmup
    elif epoch < warmup_epochs + ramp_epochs:
        return (epoch - warmup_epochs) / ramp_epochs  # Gradually ramp up KD
    else:
        return 1.0  # Full KD after ramp-up


# Function that computes the feature alignment loss by calculating the MSE, cosine and constrastive losses
def train_step(teacher_model, student_model, dataloader, optimizer, criterion, device, 
               epoch, alpha=0.5, temperature=2.0, contrastive_weight=0.5, map_loss_weight=0.3):
    teacher_model.eval()
    student_model.train()
    
    epoch_loss = 0.0
    results = {'jaccard': 0.0, 'dice': 0.0, 'recall': 0.0, 'precision': 0.0}

    curriculum_factor = dynamic_curriculum(epoch)

    for batched_images, batched_masks in dataloader:
        batched_images = batched_images.to(device, dtype=torch.float32, non_blocking=True)
        batched_masks = batched_masks.to(device, dtype=torch.float32, non_blocking=True)

        optimizer.zero_grad()

        # Pass the batch through the teacher model
        with torch.no_grad():
            _, teacher_features = teacher_model(batched_images, dataset_ids=None, weighting_mode=None, return_features=False, return_features_unet=True)
        
        student_output, student_features = student_model(batched_images, return_features=True)
        segmentation_loss = criterion(student_output, batched_masks)
        
        kd_contrastive_loss = contrastive_loss(student_features, teacher_features, temperature=temperature)
        feature_alignment_loss = compute_feature_alignment_loss(student_features, teacher_features, device)
        similarity_loss = cosine_similarity_loss(student_features, teacher_features, device)

        # Combine the segmentation loss and the feature alignment loss, perform backpropagation and update the model parameters
        total_loss = (alpha * segmentation_loss +
                    curriculum_factor * contrastive_weight * kd_contrastive_loss +
                    curriculum_factor * map_loss_weight * feature_alignment_loss +
                    curriculum_factor * 0.1 * similarity_loss)
        #print(f'Segmentation Loss: {segmentation_loss.item():.4f}, Feature Alignment Loss: {feature_alignment_loss.item():.4f}, Total Loss: {total_loss.item():.4f}')
        total_loss.backward()
        optimizer.step()

        epoch_loss += total_loss.item() * batched_images.size(0)

        for yt, yp in zip(batched_masks, student_output):
            update_metrics(results, yt, yp)

    return compute_final_results(epoch_loss, results, len(dataloader.dataset))


# Validation monitors segmentation performance only, without feature alignment loss
def evaluate_step(student_model, dataloader, criterion, device):
    student_model.eval()

    epoch_loss = 0.0
    results = {'jaccard': 0.0, 'dice': 0.0, 'recall': 0.0, 'precision': 0.0}

    with torch.inference_mode():
        for batched_images, batched_masks in dataloader:
            batched_images = batched_images.to(device, dtype=torch.float32, non_blocking=True)
            batched_masks = batched_masks.to(device, dtype=torch.float32, non_blocking=True)

            y_pred = student_model(batched_images)
            segmentation_loss = criterion(y_pred, batched_masks)

            epoch_loss += segmentation_loss.item() * batched_images.size(0)

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
        #A.CoarseDropout(p=0.3, num_holes_range=(1, 10), hole_height_range=(1, 32), hole_width_range=(1, 32))
    ])

    # Create datasets for training and validation
    train_dataset = SegmentationDataset(train_images_paths, train_masks_paths, HYPERPARAMETERS['image_size'], transform=augmentation)
    validation_dataset = SegmentationDataset(validation_images_paths, validation_masks_paths, HYPERPARAMETERS['image_size'], transform=None)

    # Create dataloaders for training and validation datasets
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=HYPERPARAMETERS['batch_size'], shuffle=True, num_workers=2, pin_memory=True, persistent_workers=True)
    validation_dataloader = DataLoader(dataset=validation_dataset, batch_size=HYPERPARAMETERS['batch_size'], shuffle=False, num_workers=2, pin_memory=True, persistent_workers=True)

    # Load dataset specific models checkpoints for each dataset
    dataset_specific_models = {dataset_id: TResUnet().to(DEVICE) for dataset_id in IDS_TO_DATASETS.keys()}
    dataset_specific_models = load_dataset_specific_models(DATASET_SPECIFIC_MODELS_CHECKPOINTS, dataset_specific_models, DEVICE)

    # Load the fused model checkpoint and create the teacher model for knowledge distillation    
    teacher_model = TResUnetFusedModel(list(dataset_specific_models.values())).to(DEVICE)
    teacher_model.load_state_dict(torch.load(FUSED_MODEL_CHECKPOINT_PATH, map_location=DEVICE))#, strict=False)
    '''incompatible = teacher_model.load_state_dict( torch.load(FUSED_MODEL_CHECKPOINT_PATH, map_location=DEVICE))#, strict=False )
    print(incompatible.missing_keys)
    print(incompatible.unexpected_keys)'''
    teacher_model = freeze_model_parameters(teacher_model)

    # Create model, optimizer, scheduler and criterion
    student_model = UNet(3, 1, True).to(DEVICE)
    optimizer = torch.optim.Adam(student_model.parameters(), lr=HYPERPARAMETERS['init_learning_rate'])
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=HYPERPARAMETERS['scheduler_patience'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=HYPERPARAMETERS['num_epochs'])
    criterion = DiceBCELoss()

    # Initialize variables for tracking the best validation metric and early stopping
    best_validation_metric = -1.0
    num_epochs_no_improvement = 0

    for epoch in range(HYPERPARAMETERS['num_epochs']):
        start_time = time.time()

        # Dynamic curriculum scheduling
        temperature = max(0.5, HYPERPARAMETERS['initial_temperature'] * (0.9 ** (epoch // 5)))
        contrastive_weight = min(HYPERPARAMETERS['max_contrastive_weight'], HYPERPARAMETERS['initial_contrastive_weight'] + epoch * HYPERPARAMETERS['weight_increment'])

        # Train and evaluate for one epoch
        train_loss, train_metrics = train_step(teacher_model, student_model, train_dataloader, optimizer, criterion, DEVICE, epoch, alpha=HYPERPARAMETERS['alpha'], temperature=temperature, contrastive_weight=contrastive_weight, map_loss_weight=HYPERPARAMETERS['map_loss_weight'])
        validation_loss, validation_metrics = evaluate_step(student_model, validation_dataloader, criterion, DEVICE)
        #scheduler.step(validation_loss)
        scheduler.step()

        # If the validation Dice (F1) score improved, save the model checkpoint and reset the early stopping counter
        if validation_metrics[1] > best_validation_metric:
            data_str = f'Valid F1 improved from {best_validation_metric:2.4f} to {validation_metrics[1]:2.4f}. Saving checkpoint: {CHECKPOINT_PATH}'
            print_and_save(TRAIN_LOG_PATH, data_str)

            best_validation_metric = validation_metrics[1]
            torch.save(student_model.state_dict(), CHECKPOINT_PATH)
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
    # Load best model and check its performance
    student_model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))

    # Load the images and masks file names for the test split
    test_images_paths, test_masks_paths = load_split_data(DATASET_PATH, 'test.txt')
    dataset_log_text = f'Test set size: {len(test_images_paths)}\n'
    print_and_save(TEST_LOG_PATH, dataset_log_text)

    # Create dataset and dataloader for the test set of the current dataset
    test_dataset = SegmentationDataset(test_images_paths, test_masks_paths, HYPERPARAMETERS['image_size'], transform=None)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=HYPERPARAMETERS['batch_size'], shuffle=False, num_workers=0, pin_memory=True)

    # Test the model
    test_loss, test_metrics = evaluate_step(student_model, test_dataloader, criterion, DEVICE)
    log_results_test(TEST_LOG_PATH, test_loss, test_metrics)