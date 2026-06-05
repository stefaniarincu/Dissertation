import os
import time
import torch
from torch.utils.data import DataLoader
import albumentations as A
from utils import seed_all, create_log_file, print_and_save, log_results_test, freeze_model_parameters, load_dataset_specific_models
from data import load_split_data, shuffle_data, SegmentationDataset, SegmentationDatasetWithDatasetId
from metrics import DiceBCELoss, update_metrics, compute_final_results
from models_tresunet import TResUnet, TResUnetFusedModel
from model_unet import UNet
from model_segformer import Segformer

SEED = 42
DEVICE = torch.device('cuda')

# Constant for hyperparameters (moved here for clarity and easy modification)
HYPERPARAMETERS = {
    'image_size': (256, 256),
    'batch_size': 16,
    'segformer_model_name': 'nvidia/mit-b2' # 'nvidia/mit-b0', 'nvidia/mit-b2', 'nvidia/mit-b4'
}

#IDS_TO_DATASETS = {0: 'bmshare', 1: 'brats', 2: 'brats_ped', 3: 'isles'}
#IDS_TO_DATASETS = {0: 'bmshare', 1: 'brats', 2: 'isles'}
#IDS_TO_DATASETS = {0: 'isles', 1: 'bmshare', 2: 'brats'}
#IDS_TO_DATASETS = {0: 'isles', 1: 'bmshare', 2: 'brats', 3: 'brats_ped'}
IDS_TO_DATASETS = {0: 'kits', 1: 'lits', 2: 'lung'}

# Constant for the root path for all necessary files
ROOT_PATH = '/root/Disertation'
EXPERIMENTS_ROOT_PATH = f'{ROOT_PATH}/files/final_experiments'

# Constants for paths of all datasets
DATASETS_ROOT_PATH = f'{ROOT_PATH}/datasets'
DATASETS_PATHS = {dataset_id: os.path.join(DATASETS_ROOT_PATH, dataset_name) for dataset_id, dataset_name in IDS_TO_DATASETS.items()}

# Constant for model checkpoint path and log path
#MODELS_AND_LOGS_ROOT_PATH = f'{EXPERIMENTS_ROOT_PATH}/dataset_specific/fract/segformer/b2'
MODELS_AND_LOGS_ROOT_PATH = f'{EXPERIMENTS_ROOT_PATH}/knowledge_distillation/kits_lits_lung/fused_from_segformers_b2_not_weighted_no_cross_attention_3K_samples_3ds_new_dropout/3_feat_student_to_teacher'

"""DATASET_SPECIFIC_MODELS_ROOT_PATH = f'{EXPERIMENTS_ROOT_PATH}/dataset_specific/fract/segformer/b2/'
DATASET_SPECIFIC_MODELS_CHECKPOINTS = {dataset_id: os.path.join(DATASET_SPECIFIC_MODELS_ROOT_PATH, dataset_name, f'dataset_specific_model_{dataset_name}.pth') for dataset_id, dataset_name in IDS_TO_DATASETS.items()}
MODEL_AND_LOG_ROOT_PATH = f'{EXPERIMENTS_ROOT_PATH}/fused/kits_lits_lung/from_segformers_b2_not_weighted_no_cross_attention_3K_samples_3ds_new_dropout_new_adapter'
CHECKPOINT_PATH = f'{MODEL_AND_LOG_ROOT_PATH}/fused_model.pth'
#MODEL_AND_LOG_ROOT_PATH = f'{EXPERIMENTS_ROOT_PATH}/knowledge_distillation/kits_lits_lung/from_fused_not_weighted_no_cross_attention_3K_samples_3ds_new_dropout/{DATASET_NAME}'
#CHECKPOINT_PATH = f'{MODEL_AND_LOG_ROOT_PATH}/distilled_model_{DATASET_NAME}.pth'
TEST_LOG_PATH = f'{MODEL_AND_LOG_ROOT_PATH}/test_log_fused.txt'"""
#LOG = f'{MODELS_AND_LOGS_ROOT_PATH}/mean_results_bmshare_brats_isles.txt'
LOG = f'{MODELS_AND_LOGS_ROOT_PATH}/mean_results_kits_lits_lung.txt'


def evaluate_step(model, dataloader, criterion, device):
    model.eval()

    epoch_loss = 0.0
    results = {'jaccard': 0.0, 'dice': 0.0, 'recall': 0.0, 'precision': 0.0, 'hd95': 0.0}

    with torch.inference_mode():
        for batched_images, batched_masks in dataloader:
            batched_images = batched_images.to(device, dtype=torch.float32, non_blocking=True)
            batched_masks = batched_masks.to(device, dtype=torch.float32, non_blocking=True)

            y_pred = model(batched_images)
            loss = criterion(y_pred, batched_masks)

            epoch_loss += loss.item() * batched_images.size(0)

            for yt, yp in zip(batched_masks, y_pred):
                update_metrics(results, yt, yp, test_mode=True)

    return compute_final_results(epoch_loss, results, len(dataloader.dataset), test_mode=True)


if __name__ == '__main__':
    seed_all(SEED)

    # Create model, optimizer, scheduler and criterion
    #model = TResUnet().to(DEVICE)
    #model = UNet(3, 1, True).to(DEVICE)
    model = Segformer.load_from_pretrained(HYPERPARAMETERS['segformer_model_name']).to(DEVICE)
    criterion = DiceBCELoss()
    
    results = {'dice': 0.0, 'jaccard': 0.0, 'recall': 0.0, 'precision': 0.0, 'hd95': 0.0}
    results_formatted = 'student'

    # Iterate through all datasets and evaluate the best model on the test set of each dataset separately, logging the results
    for dataset_id, dataset_path in DATASETS_PATHS.items():
        TEST_LOG_PATH = f'{MODELS_AND_LOGS_ROOT_PATH}/{IDS_TO_DATASETS[dataset_id]}/test_log_{IDS_TO_DATASETS[dataset_id]}.txt'
        create_log_file(TEST_LOG_PATH)

        #CHECKPOINT_PATH = f'{MODELS_AND_LOGS_ROOT_PATH}/{IDS_TO_DATASETS[dataset_id]}/dataset_specific_model_{IDS_TO_DATASETS[dataset_id]}.pth'
        CHECKPOINT_PATH = f'{MODELS_AND_LOGS_ROOT_PATH}/{IDS_TO_DATASETS[dataset_id]}/distilled_model_{IDS_TO_DATASETS[dataset_id]}.pth'
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
        model = freeze_model_parameters(model)

        # Load the images and masks file names for the test split
        test_images_paths, test_masks_paths = load_split_data(dataset_path, 'test.txt')
        #test_dataset_ids = [dataset_id] * len(test_images_paths)
        dataset_log_text = f'Test set size: {len(test_images_paths)}\n'
        print_and_save(TEST_LOG_PATH, dataset_log_text)

        # Create dataset and dataloader for the test set of the current dataset
        #test_dataset = SegmentationDatasetWithDatasetId(test_images_paths, test_masks_paths, test_dataset_ids, HYPERPARAMETERS['image_size'], transform=None)
        test_dataset = SegmentationDataset(test_images_paths, test_masks_paths, HYPERPARAMETERS['image_size'], transform=None)
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=HYPERPARAMETERS['batch_size'], shuffle=False, num_workers=0, pin_memory=True)
    
        # Test the model
        test_loss, test_metrics = evaluate_step(model, test_dataloader, criterion, DEVICE)
        log_results_test(TEST_LOG_PATH, test_loss, test_metrics, test_mode=True)

        results['dice'] += test_metrics[0] * 100.0
        results['jaccard'] += test_metrics[1] * 100.0
        results['recall'] += test_metrics[2] * 100.0
        results['precision'] += test_metrics[3] * 100.0
        results['hd95'] += test_metrics[4]
        print(f'{test_metrics[0]*100.0:.2f} & {test_metrics[1]*100.0:.2f} & {test_metrics[2]*100.0:.2f} & {test_metrics[3]*100.0:.2f} & {test_metrics[4]:.2f}\n')
        results_formatted += f' & {test_metrics[0]*100.0:.2f} & {test_metrics[1]*100.0:.2f} & {test_metrics[2]*100.0:.2f} & {test_metrics[3]*100.0:.2f} & {test_metrics[4]:.2f}'

    num_datasets = len(DATASETS_PATHS)
    results = {key: value / num_datasets for key, value in results.items()}
    print(f'Dice: {results["dice"]:.2f} - Jaccard: {results["jaccard"]:.2f} - Recall: {results["recall"]:.2f} - Precision: {results["precision"]:.2f} - HD95: {results["hd95"]:.2f}\n')
    print(f'{results["dice"]:.2f} & {results["jaccard"]:.2f} & {results["recall"]:.2f} & {results["precision"]:.2f} & {results["hd95"]:.2f}\n')
    with open(LOG, 'w') as f:
        f.write(f'Dice: {results["dice"]:.2f} - Jaccard: {results["jaccard"]:.2f} - Recall: {results["recall"]:.2f} - Precision: {results["precision"]:.2f} - HD95: {results["hd95"]:.2f}')

    print(results_formatted)

"""if __name__ == '__main__':
    seed_all(SEED)
    create_log_file(TEST_LOG_PATH)

    #dataset_specific_models = {dataset_id: TResUnet().to(DEVICE) for dataset_id in IDS_TO_DATASETS.keys()}
    #dataset_specific_models = {dataset_id: UNet(3, 1, True).to(DEVICE) for dataset_id in IDS_TO_DATASETS.keys()}
    dataset_specific_models = {dataset_id: Segformer.load_from_pretrained(HYPERPARAMETERS['segformer_model_name'], num_labels=1).to(DEVICE) for dataset_id in IDS_TO_DATASETS.keys()}
    dataset_specific_models = load_dataset_specific_models(DATASET_SPECIFIC_MODELS_CHECKPOINTS, dataset_specific_models, DEVICE)
    #model = TResUnetFusedModel(list(dataset_specific_models.values())).to(DEVICE)
    #model = TResUnetFusedModel(list(dataset_specific_models.values()), use_unets=True).to(DEVICE)
    model = TResUnetFusedModel(list(dataset_specific_models.values()), use_segformers=True).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model = freeze_model_parameters(model)

    criterion = DiceBCELoss()
    
    results = {'dice': 0.0, 'jaccard': 0.0, 'recall': 0.0, 'precision': 0.0, 'hd95': 0.0}
    results_as_dict = {0: [], 1: [], 2: [], 3: []}

    # Iterate through all datasets and evaluate the best model on the test set of each dataset separately, logging the results
    for dataset_id, dataset_path in DATASETS_PATHS.items():
        dataset_log_text = f'{IDS_TO_DATASETS[dataset_id]} dataset'
        print_and_save(TEST_LOG_PATH, dataset_log_text)

        # Load the images and masks file names for the test split
        test_images_paths, test_masks_paths = load_split_data(dataset_path, 'test.txt')
        #test_dataset_ids = [dataset_id] * len(test_images_paths)
        dataset_log_text = f'Test set size: {len(test_images_paths)}\n'
        print_and_save(TEST_LOG_PATH, dataset_log_text)

        # Create dataset and dataloader for the test set of the current dataset
        #test_dataset = SegmentationDatasetWithDatasetId(test_images_paths, test_masks_paths, test_dataset_ids, HYPERPARAMETERS['image_size'], transform=None)
        test_dataset = SegmentationDataset(test_images_paths, test_masks_paths, HYPERPARAMETERS['image_size'], transform=None)
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=HYPERPARAMETERS['batch_size'], shuffle=False, num_workers=0, pin_memory=True)
    
        # Test the model
        test_loss, test_metrics = evaluate_step(model, test_dataloader, criterion, DEVICE)
        log_results_test(TEST_LOG_PATH, test_loss, test_metrics, test_mode=True)

        results['dice'] += test_metrics[0] * 100.0
        results['jaccard'] += test_metrics[1] * 100.0
        results['recall'] += test_metrics[2] * 100.0
        results['precision'] += test_metrics[3] * 100.0
        results['hd95'] += test_metrics[4]
        print(f'{test_metrics[0]*100.0:.2f} & {test_metrics[1]*100.0:.2f} & {test_metrics[2]*100.0:.2f} & {test_metrics[3]*100.0:.2f} & {test_metrics[4]:.2f}\n')
        results_as_dict[dataset_id] = test_metrics

    num_datasets = len(DATASETS_PATHS)
    results = {key: value / num_datasets for key, value in results.items()}
    print(f'Dice: {results["dice"]:.2f} - Jaccard: {results["jaccard"]:.2f} - Recall: {results["recall"]:.2f} - Precision: {results["precision"]:.2f} - HD95: {results["hd95"]:.2f}\n')
    print(f'{results["dice"]:.2f} & {results["jaccard"]:.2f} & {results["recall"]:.2f} & {results["precision"]:.2f} & {results["hd95"]:.2f}\n')
    with open(LOG, 'w') as f:
        f.write(f'Dice: {results["dice"]:.2f} - Jaccard: {results["jaccard"]:.2f} - Recall: {results["recall"]:.2f} - Precision: {results["precision"]:.2f} - HD95: {results["hd95"]:.2f}')

    results_formatted = 'fused teacher'
    '''results_formatted += f' & {results_as_dict[1][0]*100.0:.2f} & {results_as_dict[1][1]*100.0:.2f} & {results_as_dict[1][2]*100.0:.2f} & {results_as_dict[1][3]*100.0:.2f} & {results_as_dict[1][4]:.2f}'
    results_formatted += f' & {results_as_dict[2][0]*100.0:.2f} & {results_as_dict[2][1]*100.0:.2f} & {results_as_dict[2][2]*100.0:.2f} & {results_as_dict[2][3]*100.0:.2f} & {results_as_dict[2][4]:.2f}'
    #results_formatted += f' & {results_as_dict[3][0]*100.0:.2f} & {results_as_dict[3][1]*100.0:.2f} & {results_as_dict[3][2]*100.0:.2f} & {results_as_dict[3][3]*100.0:.2f} & {results_as_dict[3][4]:.2f}'
    results_formatted += f' & {results_as_dict[0][0]*100.0:.2f} & {results_as_dict[0][1]*100.0:.2f} & {results_as_dict[0][2]*100.0:.2f} & {results_as_dict[0][3]*100.0:.2f} & {results_as_dict[0][4]:.2f}'
    print(results_formatted)'''
    results_formatted += f' & {results_as_dict[0][0]*100.0:.2f} & {results_as_dict[0][1]*100.0:.2f} & {results_as_dict[0][2]*100.0:.2f} & {results_as_dict[0][3]*100.0:.2f} & {results_as_dict[0][4]:.2f}'
    results_formatted += f' & {results_as_dict[1][0]*100.0:.2f} & {results_as_dict[1][1]*100.0:.2f} & {results_as_dict[1][2]*100.0:.2f} & {results_as_dict[1][3]*100.0:.2f} & {results_as_dict[1][4]:.2f}'
    results_formatted += f' & {results_as_dict[2][0]*100.0:.2f} & {results_as_dict[2][1]*100.0:.2f} & {results_as_dict[2][2]*100.0:.2f} & {results_as_dict[2][3]*100.0:.2f} & {results_as_dict[2][4]:.2f}'
    print(results_formatted)"""