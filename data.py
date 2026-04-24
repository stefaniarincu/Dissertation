import os
import cv2 as cv
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
from sklearn.utils import shuffle
import random


''' ============================================== LOADING DATA ============================================== '''
 
# Function that loads a specified split data from specified path
def load_split_data(dataset_path, split_filename):
    split_path = os.path.join(dataset_path, split_filename)
    with open(split_path, 'r') as f:
        filenames = [line.strip() for line in f if line.strip()]

    images_paths = [os.path.join(dataset_path, 'images', name) for name in filenames]
    masks_paths = [os.path.join(dataset_path, 'masks', name) for name in filenames]
    
    return images_paths, masks_paths

# Function that loads data from all datasets
def load_split_data_all_datasets(datasets_paths, split_filename, num_train_samples_per_dataset=None, num_val_samples_per_dataset=None):
    images_by_dataset_id, masks_by_dataset_id = {}, {}
    # Collect the images and masks paths for each dataset and store them in a dictionary that maps dataset ids to the corresponding paths
    for dataset_id, dataset_path in datasets_paths.items():
        images_paths, masks_paths = load_split_data(dataset_path, split_filename)
        images_by_dataset_id[dataset_id] = images_paths
        masks_by_dataset_id[dataset_id] = masks_paths

    all_images, all_masks, all_dataset_ids = [], [], []
    for dataset_id in images_by_dataset_id.keys():
        all_images.extend(images_by_dataset_id[dataset_id])
        all_masks.extend(masks_by_dataset_id[dataset_id])
        all_dataset_ids.extend([dataset_id] * len(images_by_dataset_id[dataset_id]))

    '''# Take the number of required samples from each dataset if specified, otherwise take the minimum number of samples available across all datasets 
    if split_filename == 'train.txt':
        if num_train_samples_per_dataset is not None:
            min_size = num_train_samples_per_dataset
        else:
            min_size = min(len(images_by_dataset_id[dataset_id]) for dataset_id in images_by_dataset_id.keys())
    else:
        if num_val_samples_per_dataset is not None:
            min_size = num_val_samples_per_dataset
        else:
            min_size = min(len(images_by_dataset_id[dataset_id]) for dataset_id in images_by_dataset_id.keys())
    
    all_images, all_masks, all_dataset_ids = [], [], []
    for dataset_id in images_by_dataset_id.keys():
        all_images.extend(images_by_dataset_id[dataset_id][:min_size])
        all_masks.extend(masks_by_dataset_id[dataset_id][:min_size])
        all_dataset_ids.extend([dataset_id] * min_size)'''
    
    return all_images, all_masks, all_dataset_ids


''' ============================================= SHUFFLING DATA ============================================= '''

# Function that shuffles the given data (images and masks and in some cases ids)
def shuffle_data(data, random_state):
    # If the length of the data is 2, then unpack the images and masks paths and shuffle them
    if len(data) == 2:
        images_paths, masks_paths = data
        return shuffle(images_paths, masks_paths, random_state=random_state)
    # If the length of the data is 3, then unpack the images and masks paths and ids and shuffle them
    else:
        images_paths, masks_paths, datasets_ids = data
        return shuffle(images_paths, masks_paths, datasets_ids, random_state=random_state)
    

''' ============================================ DATASET CLASSES ============================================ '''

def pad_black_to_256(image, mask):
    width, height = image.shape[1], image.shape[0]
    pad_width, pad_height = 256 - width, 256 - height
    
    top = pad_height // 2
    bottom = pad_height - top
    
    left = pad_width // 2
    right = pad_width - left

    padded_image = cv.copyMakeBorder(image, top, bottom, left, right, cv.BORDER_CONSTANT, value=[0, 0, 0])
    padded_mask = cv.copyMakeBorder(mask, top, bottom, left, right, cv.BORDER_CONSTANT, value=0)
    return padded_image, padded_mask

# Base segmentation Dataset class for loading images and masks
class BaseSegmentationDataset(Dataset):
    def __init__(self, images_paths, masks_paths, image_size, transform=None):
        super().__init__()

        self.images_paths = images_paths
        self.masks_paths = masks_paths
        self.image_size = image_size
        self.transform = transform

    def __len__(self):
        return len(self.images_paths)

    def load_sample(self, index):
        image = cv.imread(self.images_paths[index], cv.IMREAD_COLOR)
        mask = cv.imread(self.masks_paths[index], cv.IMREAD_GRAYSCALE)

        '''if image.shape[0] < 256 or image.shape[1] < 256:
            image, mask = pad_black_to_256(image, mask)'''
        
        '''if image.shape[0] != 256 or  image.shape[1] != 256:
            image = cv.resize(image, self.image_size, interpolation=cv.INTER_LINEAR)
            mask = cv.resize(mask, self.image_size, interpolation=cv.INTER_NEAREST)'''

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations['image']
            mask = augmentations['mask']

        image = cv.resize(image, self.image_size, interpolation=cv.INTER_LINEAR)
        mask = cv.resize(mask, self.image_size, interpolation=cv.INTER_NEAREST)

        image = torch.from_numpy(image).permute(2, 0, 1).float()
        image.div_(255.0)

        mask = (mask > 127).astype(np.float32)
        mask = torch.from_numpy(mask).unsqueeze(0).float()

        return image, mask

# Class that inherits from the BaseSegmentationDataset and is used for loading images and masks without dataset ids
class SegmentationDataset(BaseSegmentationDataset):
    def __getitem__(self, index):
        image, mask = self.load_sample(index)
        return image, mask
    
# Class that inherits from the BaseSegmentationDataset and is used for loading images and masks with dataset ids
class SegmentationDatasetWithDatasetId(BaseSegmentationDataset):
    def __init__(self, images_paths, masks_paths, dataset_ids, image_size, transform=None): 
        super().__init__(images_paths, masks_paths, image_size, transform)
        self.dataset_ids = dataset_ids 
    
    def __getitem__(self, index): 
        image, mask = self.load_sample(index) 
        dataset_id = torch.tensor(self.dataset_ids[index], dtype=torch.long) 
        return image, mask, dataset_id
    

''' ========================================= BALANCED BATCH SAMPLING ========================================= '''
    
# Custom class for balanced batch sampling that ensures that each batch contains a balanced number of samples from each dataset
class BalancedBatchSampler(Sampler):
    '''
    For batch_size=16 and 3 datasets:
        batch 1 -> 6, 5, 5 samples from dataset 1, 2, 3
        batch 2 -> 5, 6, 5 samples from dataset 1, 2, 3
        batch 3 -> 5, 5, 6 samples from dataset 1, 2, 3
        and so on

    For batch_size=16 and 4 datasets:
        all batches -> 4 samples from each dataset
    '''
    def __init__(self, dataset_ids, batch_size, seed, shuffle=True, allow_incomplete_last_batch=False):
        super().__init__()
        self.dataset_ids = np.array(dataset_ids, dtype=np.int64)
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.allow_incomplete_last_batch = allow_incomplete_last_batch
        self.epoch = 0

        self.unique_dataset_ids = sorted(np.unique(self.dataset_ids).tolist())
        self.indices_per_dataset = {dataset_id: np.where(self.dataset_ids == dataset_id)[0].tolist() for dataset_id in self.unique_dataset_ids}
        
        # Minimum number of samples each dataset contributes to every batch
        self.base_count = self.batch_size // len(self.indices_per_dataset)

        # Remaining samples are rotated across datasets from batch to batch
        self.remainder = self.batch_size % len(self.indices_per_dataset)
        
        if self.allow_incomplete_last_batch:
            self.num_batches = int(np.ceil(len(self.dataset_ids) / self.batch_size))
        else:
            self.num_batches = len(self.dataset_ids) // self.batch_size

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.num_batches
    
    def __iter__(self):
        # Create a deterministic random generator for shuffling indices
        if self.shuffle:
            random_generator = random.Random(self.seed + self.epoch)

        datasets_indices, current_indices = {}, {}
        for dataset_id in self.unique_dataset_ids:
            dataset_indices = self.indices_per_dataset[dataset_id].copy()
            
            if self.shuffle:
                random_generator.shuffle(dataset_indices)
            
            datasets_indices[dataset_id] = dataset_indices
            current_indices[dataset_id] = 0

        for batch_index in range(self.num_batches):
            sample_counts_per_dataset = {dataset_id: self.base_count for dataset_id in self.unique_dataset_ids}
            
            # Distrbute the remaining samples by rotating which datasets receive one extra sample
            for extra_index in range(self.remainder):
                dataset_id = self.unique_dataset_ids[(batch_index + extra_index + self.epoch) % len(self.unique_dataset_ids)]
                sample_counts_per_dataset[dataset_id] += 1

            batch = []

            for dataset_id in self.unique_dataset_ids:
                num_required_samples = sample_counts_per_dataset[dataset_id]
                dataset_indices = datasets_indices[dataset_id]

                start_index = current_indices[dataset_id]
                num_remaining_samples = len(dataset_indices) - start_index
                num_samples_to_take = min(num_required_samples, num_remaining_samples)

                if num_samples_to_take > 0:
                    end_index = start_index + num_samples_to_take
                    batch.extend(dataset_indices[start_index:end_index])
                    current_indices[dataset_id] = end_index

            num_missing_samples = self.batch_size - len(batch)
            
            # If full batches are required, try to fill the missing positions with any remaining samples
            if num_missing_samples > 0 and not self.allow_incomplete_last_batch:
                for dataset_id in self.unique_dataset_ids:
                    if num_missing_samples == 0:
                        break

                    dataset_indices = datasets_indices[dataset_id]
                    start_index = current_indices[dataset_id]
                    num_remaining_samples = len(dataset_indices) - start_index

                    if num_remaining_samples > 0:
                        num_samples_to_take = min(num_missing_samples, num_remaining_samples)
                        end_index = start_index + num_samples_to_take
                        batch.extend(dataset_indices[start_index:end_index])
                        current_indices[dataset_id] = end_index
                        num_missing_samples -= num_samples_to_take

                if len(batch) < self.batch_size:
                    return

            if len(batch) == 0:
                return

            if self.shuffle:
                random_generator.shuffle(batch)

            yield batch

class BalancedBatchSamplerWithCustomComposition(Sampler):
    def __init__(self, dataset_ids, seed, batch_composition={0: 4, 1: 4, 2: 8}):
        super().__init__()

        self.dataset_ids = np.array(dataset_ids, dtype=np.int64)
        self.batch_composition = batch_composition
        self.epoch = 0
        self.seed = seed

        self.ids_indices = {dataset_id: np.where(self.dataset_ids == dataset_id)[0] for dataset_id in self.batch_composition.keys()}

        batches_per_dataset = []
        for dataset_id in self.batch_composition.keys():
            num_samples = len(self.ids_indices[dataset_id])
            num_batches = int(np.ceil(num_samples / self.batch_composition[dataset_id]))
            batches_per_dataset.append(num_batches)

        self.num_batches = max(batches_per_dataset)

    def set_epoch(self, param_epoch):
        self.epoch = param_epoch

    def __len__(self):
        return self.num_batches
    
    def __iter__(self):
        random_generator = random.Random(self.seed + self.epoch)
        
        shuffled_indices, current_indices = {}, {}
        for dataset_id in self.batch_composition.keys():
            dataset_indices = self.ids_indices[dataset_id].copy()
            random_generator.shuffle(dataset_indices)
            shuffled_indices[dataset_id] = dataset_indices
            current_indices[dataset_id] = 0

        for _ in range(self.num_batches):
            batch = []

            for dataset_id, num_samples_per_batch in self.batch_composition.items():
                start_idx = current_indices[dataset_id]
                end_idx = start_idx + num_samples_per_batch

                if end_idx > len(shuffled_indices[dataset_id]):
                    random_generator.shuffle(shuffled_indices[dataset_id])
                    current_indices[dataset_id] = 0
                    start_idx = 0
                    end_idx = num_samples_per_batch

                batch.extend(shuffled_indices[dataset_id][start_idx:end_idx])
                current_indices[dataset_id] = end_idx

            random_generator.shuffle(batch)
            yield batch