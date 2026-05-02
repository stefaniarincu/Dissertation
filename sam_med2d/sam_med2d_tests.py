import os
import time
import torch
from torch.utils.data import DataLoader
import albumentations as A
from utils import seed_all, create_log_file, print_and_save, log_hyperparameters, log_results_train_val, log_results_test
from data import load_split_data, shuffle_data, SegmentationDataset
from metrics import DiceBCELoss, update_metrics, compute_final_results
from models_tresunet import TResUnet

import sys
sys.path.append('/root/Disertation/sam_med2d')
from utils_file import get_boxes_from_mask, FocalDiceloss_IoULoss
from segment_anything import sam_model_registry
from torch.nn import functional as F

SEED = 42
DEVICE = torch.device('cuda')

# Constant for hyperparameters (moved here for clarity and easy modification)
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
DATASET_NAME = 'isles' # 'isles', 'bmshare', 'brats', 'brats_ped'
DATASET_PATH = f'{ROOT_PATH}/datasets/{DATASET_NAME}'

# Constant for sammed2d model checkpoint path
PRETRAINED_SAM_MED2D_CHECKPOINT_PATH = f'{ROOT_PATH}/sam_med2d/pretrained_checkpoint/sam-med2d_b.pth'
FINE_TUNED_SAM_MED2D_PATH = f'{ROOT_PATH}/files/dataset_specific/sam_med2d/dice/{DATASET_NAME}'
FINE_TUNED_SAM_MED2D_CHECKPOINT_PATH = f'{FINE_TUNED_SAM_MED2D_PATH}/dataset_specific_model_{DATASET_NAME}.pth'

# Constant for tresunet model checkpoint path
TRESUNET_PATH = f'{ROOT_PATH}/files/dataset_specific/{DATASET_NAME}'
TRESUNET_CHECKPOINT_PATH = f'{TRESUNET_PATH}/dataset_specific_model_{DATASET_NAME}.pth'

# Log path
LOG_PATH = f'{ROOT_PATH}/files/dataset_specific/sam_med2d/comparison'
os.makedirs(LOG_PATH, exist_ok=True)
LOG_FILE_PATH = f'{LOG_PATH}/comparison_{DATASET_NAME}.txt'

def sam_forward(model, image, batched_masks, use_mask_as_prompt=True):
    image_embeddings = model.image_encoder(image)

    if use_mask_as_prompt:
        boxes = torch.stack([get_boxes_from_mask(mask.squeeze().cpu().numpy()) for mask in batched_masks])
        boxes = boxes.to(DEVICE, dtype=torch.float32)
        sparse_emb, dense_emb = model.prompt_encoder(points=None, boxes=boxes, masks=None)
    else:
        sparse_emb, dense_emb = model.prompt_encoder(points=None, boxes=None, masks=None)

    low_res_masks, iou_predictions = model.mask_decoder(
        image_embeddings=image_embeddings,
        image_pe=model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_emb,
        dense_prompt_embeddings=dense_emb,
        multimask_output=False
    )

    masks = F.interpolate(low_res_masks, size=image.shape[2:], mode='bilinear', align_corners=False)
    return masks, low_res_masks, iou_predictions

def evaluate_step_with_gt(model, dataloader, criterion, device):
    model.eval()

    epoch_loss = 0.0
    results = {'jaccard': 0.0, 'dice': 0.0, 'recall': 0.0, 'precision': 0.0}

    with torch.inference_mode():
        for batched_images, batched_masks in dataloader:
            batched_images = batched_images.to(device, dtype=torch.float32, non_blocking=True)
            batched_masks = batched_masks.to(device, dtype=torch.float32, non_blocking=True)

            masks, low_res_masks, iou_predictions = sam_forward(model, batched_images, batched_masks)
            #loss = criterion(masks, batched_masks, iou_predictions)
            loss = criterion(masks, batched_masks)

            epoch_loss += loss.item() * batched_images.size(0)

            for yt, yp in zip(batched_masks, masks):
                update_metrics(results, yt, yp)

    return compute_final_results(epoch_loss, results, len(dataloader.dataset))

def evaluate_step_with_tresunet(tresunet_model, sam_model, dataloader, criterion, device):
    tresunet_model.eval()
    sam_model.eval()

    epoch_loss = 0.0
    results = {'jaccard': 0.0, 'dice': 0.0, 'recall': 0.0, 'precision': 0.0}

    with torch.inference_mode():
        for batched_images, batched_masks in dataloader:
            batched_images = batched_images.to(device, dtype=torch.float32, non_blocking=True)
            batched_masks = batched_masks.to(device, dtype=torch.float32, non_blocking=True)

            predicted_masks = tresunet_model(batched_images)
            predicted_masks = torch.sigmoid(predicted_masks)
            predicted_masks = (predicted_masks > 0.5).float()

            masks, low_res_masks, iou_predictions = sam_forward(sam_model, batched_images, predicted_masks)
            #loss = criterion(masks, batched_masks, iou_predictions)
            loss = criterion(masks, batched_masks)

            epoch_loss += loss.item() * batched_images.size(0)

            for yt, yp in zip(batched_masks, masks):
                update_metrics(results, yt, yp)

    return compute_final_results(epoch_loss, results, len(dataloader.dataset))

def evaluate_step_with_box_as_whole_image(model, dataloader, criterion, device):
    model.eval()

    epoch_loss = 0.0
    results = {'jaccard': 0.0, 'dice': 0.0, 'recall': 0.0, 'precision': 0.0}

    with torch.inference_mode():
        for batched_images, batched_masks in dataloader:
            batched_images = batched_images.to(device, dtype=torch.float32, non_blocking=True)
            batched_masks = batched_masks.to(device, dtype=torch.float32, non_blocking=True)

            batched_masks_as_boxes = torch.ones_like(batched_masks)
            masks, low_res_masks, iou_predictions = sam_forward(model, batched_images, batched_masks_as_boxes)
            #loss = criterion(masks, batched_masks, iou_predictions)
            loss = criterion(masks, batched_masks)

            epoch_loss += loss.item() * batched_images.size(0)

            for yt, yp in zip(batched_masks, masks):
                update_metrics(results, yt, yp)

    return compute_final_results(epoch_loss, results, len(dataloader.dataset))

def evaluate_step_with_no_prompt(model, dataloader, criterion, device):
    model.eval()

    epoch_loss = 0.0
    results = {'jaccard': 0.0, 'dice': 0.0, 'recall': 0.0, 'precision': 0.0}

    with torch.inference_mode():
        for batched_images, batched_masks in dataloader:
            batched_images = batched_images.to(device, dtype=torch.float32, non_blocking=True)
            batched_masks = batched_masks.to(device, dtype=torch.float32, non_blocking=True)

            batched_masks_as_boxes = torch.ones_like(batched_masks)
            masks, low_res_masks, iou_predictions = sam_forward(model, batched_images, batched_masks_as_boxes, use_mask_as_prompt=False)
            #loss = criterion(masks, batched_masks, iou_predictions)
            loss = criterion(masks, batched_masks)

            epoch_loss += loss.item() * batched_images.size(0)

            for yt, yp in zip(batched_masks, masks):
                update_metrics(results, yt, yp)

    return compute_final_results(epoch_loss, results, len(dataloader.dataset))

if __name__ == '__main__':
    seed_all(SEED)
    create_log_file(LOG_FILE_PATH)

    # Load the images and masks file names for training and validation
    #train_images_paths, train_masks_paths = load_split_data(DATASET_PATH, 'train.txt')
    #validation_images_paths, validation_masks_paths = load_split_data(DATASET_PATH, 'val.txt')
    test_images_paths, test_masks_paths = load_split_data(DATASET_PATH, 'test.txt')
    #dataset_log_text = f'Train set size: {len(train_images_paths)}\nValidation set size: {len(validation_images_paths)}\nTest set size: {len(test_images_paths)}\n'
    dataset_log_text = f'Test set size: {len(test_images_paths)}\n'
    print_and_save(LOG_FILE_PATH, dataset_log_text)
    #print_and_save(LOG_FILE_PATH, f'Used fractions instead of absolute values for coarse dropout augmentation in training\n')

    # Create datasets for training, validation and test splits
    #train_dataset = SegmentationDataset(train_images_paths, train_masks_paths, HYPERPARAMETERS['image_size'], transform=None)
    #validation_dataset = SegmentationDataset(validation_images_paths, validation_masks_paths, HYPERPARAMETERS['image_size'], transform=None)
    test_dataset = SegmentationDataset(test_images_paths, test_masks_paths, HYPERPARAMETERS['image_size'], transform=None)
    
    # Create dataloaders for training, validation and test datasets
    #train_dataloader = DataLoader(dataset=train_dataset, batch_size=HYPERPARAMETERS['batch_size'], shuffle=False, num_workers=0, pin_memory=True)
    #validation_dataloader = DataLoader(dataset=validation_dataset, batch_size=HYPERPARAMETERS['batch_size'], shuffle=False, num_workers=0, pin_memory=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=HYPERPARAMETERS['batch_size'], shuffle=False, num_workers=0, pin_memory=True)

    # Create sam model arguments class
    class SAMArgs:
        image_size = HYPERPARAMETERS['image_size'][0]
        encoder_adapter = True
        sam_checkpoint = PRETRAINED_SAM_MED2D_CHECKPOINT_PATH

    # Load sam-med2d model and its pretrained weights
    pretrained_sam_med2d = sam_model_registry['vit_b'](SAMArgs()).to(DEVICE)
    checkpoint = torch.load(PRETRAINED_SAM_MED2D_CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    pretrained_sam_med2d.load_state_dict(checkpoint['model'])
    pretrained_sam_med2d.eval()

    # Load fine-tuned sam-med2d model and its pretrained weights
    fine_tuned_sam_med2d = sam_model_registry['vit_b'](SAMArgs()).to(DEVICE)
    #checkpoint = torch.load(PRETRAINED_SAM_MED2D_CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    #fine_tuned_sam_med2d.load_state_dict(checkpoint['model'])
    fine_tuned_sam_med2d.load_state_dict(torch.load(FINE_TUNED_SAM_MED2D_CHECKPOINT_PATH, map_location=DEVICE))
    fine_tuned_sam_med2d.eval()

    # Load tresunet model and its pretrained weights
    tresunet_model = TResUnet().to(DEVICE)
    tresunet_model.load_state_dict(torch.load(TRESUNET_CHECKPOINT_PATH, map_location=DEVICE))
    tresunet_model.eval()

    # Define the loss function
    criterion = DiceBCELoss()

    # Evaluate fine-tuned sam-med2d model with gt masks
    print_and_save(LOG_FILE_PATH, f'Fine-tuned Sam-Med2D with gt masks:')
    test_loss, test_metrics = evaluate_step_with_gt(fine_tuned_sam_med2d, test_dataloader, criterion, DEVICE)
    log_results_test(LOG_FILE_PATH, test_loss, test_metrics)

    # Evaluate fine-tuned sam-med2d model with tresunet predicted masks
    print_and_save(LOG_FILE_PATH, f'Fine-tuned Sam-Med2D with tresunet predicted masks:')
    test_loss, test_metrics = evaluate_step_with_tresunet(tresunet_model, fine_tuned_sam_med2d, test_dataloader, criterion, DEVICE)
    log_results_test(LOG_FILE_PATH, test_loss, test_metrics)

    # Evaluate fine-tuned sam-med2d model with box as whole image
    print_and_save(LOG_FILE_PATH, f'Fine-tuned Sam-Med2D with box as whole image:')
    test_loss, test_metrics = evaluate_step_with_box_as_whole_image(fine_tuned_sam_med2d, test_dataloader, criterion, DEVICE)
    log_results_test(LOG_FILE_PATH, test_loss, test_metrics)

    # Evaluate fine-tuned sam-med2d model with no prompt
    print_and_save(LOG_FILE_PATH, f'Fine-tuned Sam-Med2D with no prompt:')
    test_loss, test_metrics = evaluate_step_with_no_prompt(fine_tuned_sam_med2d, test_dataloader, criterion, DEVICE)
    log_results_test(LOG_FILE_PATH, test_loss, test_metrics)

    # Evaluate pretrained sam-med2d model with gt masks
    print_and_save(LOG_FILE_PATH, f'Pretrained Sam-Med2D with gt masks:')
    test_loss, test_metrics = evaluate_step_with_gt(pretrained_sam_med2d, test_dataloader, criterion, DEVICE)
    log_results_test(LOG_FILE_PATH, test_loss, test_metrics)

    # Evaluate pretrained sam-med2d model with tresunet predicted masks
    print_and_save(LOG_FILE_PATH, f'Pretrained Sam-Med2D with tresunet predicted masks:')
    test_loss, test_metrics = evaluate_step_with_tresunet(tresunet_model, pretrained_sam_med2d, test_dataloader, criterion, DEVICE)
    log_results_test(LOG_FILE_PATH, test_loss, test_metrics)

    # Evaluate pretrained sam-med2d model with box as whole image
    print_and_save(LOG_FILE_PATH, f'Pretrained Sam-Med2D with box as whole image:')
    test_loss, test_metrics = evaluate_step_with_box_as_whole_image(pretrained_sam_med2d, test_dataloader, criterion, DEVICE)
    log_results_test(LOG_FILE_PATH, test_loss, test_metrics)

    # Evaluate pretrained sam-med2d model with no prompt
    print_and_save(LOG_FILE_PATH, f'Pretrained Sam-Med2D with no prompt:')
    test_loss, test_metrics = evaluate_step_with_no_prompt(pretrained_sam_med2d, test_dataloader, criterion, DEVICE)
    log_results_test(LOG_FILE_PATH, test_loss, test_metrics)