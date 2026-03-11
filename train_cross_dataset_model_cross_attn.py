import os
import random
import datetime
import time
import numpy as np
import cv2 as cv
from sklearn.utils import shuffle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.hub import load_state_dict_from_url
import torch.nn.functional as F
import albumentations as A

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
    'early_stopping_patience': 25,
    'dsm_weight_mode': 2, # dataset specific models weighting - 0 one-hot (match own dataset expert), 1 uniform, 2 biasd towards own dataset expert
}

# Dictionary that maps dataset names to an id
DATASETS_TO_IDS = {'isles': 0, 'bmshare': 1, 'brats': 2}

# Constants for dataset paths
DATASETS_ROOT_PATH = '/home/dragos/disertation/datasets'
DATASETS_PATHS = {dataset_name: os.path.join(DATASETS_ROOT_PATH, dataset_name) for dataset_name in DATASETS_TO_IDS.keys()}
# Constant for dataset specific models checkpoint paths and a mapping from dataset names to paths
DATASET_SPECIFIC_MODELS_ROOT_PATH = '/home/dragos/disertation/files'
DATASET_SPECIFIC_MODELS_CHECKPOINTS = {dataset_name: os.path.join(DATASET_SPECIFIC_MODELS_ROOT_PATH, dataset_name, f'dataset_specific_model_{dataset_name}.pth') for dataset_name in DATASETS_TO_IDS.keys()}
# Constants for model checkpoint path and log path
MODELS_AND_LOG_ROOT_PATH = '/home/dragos/disertation/files/cross_dataset/biased/cross_attn'
os.makedirs(MODELS_AND_LOG_ROOT_PATH, exist_ok=True)
CHECKPOINT_PATH = f'{MODELS_AND_LOG_ROOT_PATH}/cross_dataset_model.pth'
LOG_PATH = f'{MODELS_AND_LOG_ROOT_PATH}/train_log_cross_dataset.txt'
# Constant for resume checkpoint path if the training stops for whatever reason
RESUME_CHECKPOINT_PATH = f'{MODELS_AND_LOG_ROOT_PATH}/cross_dataset_last_resume.pth'

# Function that sets constant seed for reproducibility
def seed_all(param_seed=SEED):
    random.seed(param_seed)
    os.environ['PYTHONHASHSEED'] = str(param_seed)
    np.random.seed(param_seed)
    torch.manual_seed(param_seed)
    torch.cuda.manual_seed(param_seed)
    torch.backends.cudnn.deterministic = True

# Function that prints a message and also saves it to a specified file
def print_and_save(param_file_path, param_text):
    print(param_text)
    with open(param_file_path, 'a') as file:
        file.write(param_text)
        file.write('\n')

# Function that loads all file names for images and masks in the dataset
def load_split_filenames(param_dataset_path, param_split_file):
    file_names = open(param_split_file, 'r').read().split('\n')[:-1]
    images = [os.path.join(param_dataset_path, 'images', name) for name in file_names]
    masks = [os.path.join(param_dataset_path, 'masks', name) for name in file_names]
    return images, masks

# Function that loads training and validation data from all datasets, keeping the same number of samples for each dataset by limiting to the size of the smallest dataset
def load_data(param_dataset_paths):
    train_images_by_dataset, train_masks_by_dataset = {}, {}
    validation_images_by_dataset, validation_masks_by_dataset = {}, {}

    for dataset_name, dataset_path in param_dataset_paths.items():
        train_images, train_masks = load_split_filenames(dataset_path, os.path.join(dataset_path, 'train.txt'))
        validation_images, validation_masks = load_split_filenames(dataset_path, os.path.join(dataset_path, 'val.txt'))

        dataset_id = DATASETS_TO_IDS[dataset_name]
        train_images_by_dataset[dataset_id] = train_images
        train_masks_by_dataset[dataset_id] = train_masks
        validation_images_by_dataset[dataset_id] = validation_images
        validation_masks_by_dataset[dataset_id] = validation_masks

    # Keep the same number of samples for each dataset by limiting to the size of the smallest dataset
    min_size_train = min(len(train_images_by_dataset[dataset_id]) for dataset_id in train_images_by_dataset.keys())
    all_train_images, all_train_masks, all_train_dataset_ids = [], [], []
    #print(f'Min size of training datasets: {min_size_train}')
    for dataset_id in train_images_by_dataset.keys():
        all_train_images.extend(train_images_by_dataset[dataset_id][:min_size_train])
        all_train_masks.extend(train_masks_by_dataset[dataset_id][:min_size_train])
        all_train_dataset_ids.extend([dataset_id] * min_size_train)

    min_size_validation = min(len(validation_images_by_dataset[dataset_id]) for dataset_id in validation_images_by_dataset.keys())
    all_validation_images, all_validation_masks, all_validation_dataset_ids = [], [], []
    #print(f'Min size of validation datasets: {min_size_validation}')
    for dataset_id in validation_images_by_dataset.keys():
        all_validation_images.extend(validation_images_by_dataset[dataset_id][:min_size_validation])
        all_validation_masks.extend(validation_masks_by_dataset[dataset_id][:min_size_validation])
        all_validation_dataset_ids.extend([dataset_id] * min_size_validation)

    return (all_train_images, all_train_masks, all_train_dataset_ids), (all_validation_images, all_validation_masks, all_validation_dataset_ids)

# Function that shuffles the images, masks and dataset ids
def shuffling(param_images_path, param_masks_path, param_dataset_ids):
    param_images_path, param_masks_path, param_dataset_ids = shuffle(param_images_path, param_masks_path, param_dataset_ids, random_state=SEED)
    return param_images_path, param_masks_path, param_dataset_ids

# Function that loads dataset specific models from specified checkpoints and returns a dictionary that maps dataset ids to the corresponding model
def load_dataset_specific_models(param_dataset_specific_models_checkpoints, param_device=DEVICE):
    dataset_specific_models = {}
    for dataset_name, checkpoint_path in param_dataset_specific_models_checkpoints.items():
        dataset_specific_model = TResUnet().to(param_device)
        dataset_specific_model.load_state_dict(torch.load(checkpoint_path, map_location=param_device))
        dataset_specific_model.eval()

        # Freeze the parameters of each dataset specific model
        for param in dataset_specific_model.parameters():
            param.requires_grad = False

        dataset_specific_models[DATASETS_TO_IDS[dataset_name]] = dataset_specific_model
    return dataset_specific_models

# Determine the weights for each dataset specific model based on the specified mode and the dataset ids of the samples in the batch
def determine_weights_for_dataset_specific_models(param_dataset_ids, param_mode=0, param_num_dataset_specific_models=len(DATASETS_TO_IDS)):
    if param_mode == 0: # one hot encoding = > 1 for the dataset specific model corresponding to the dataset and 0 for the others
        return F.one_hot(param_dataset_ids, num_classes=param_num_dataset_specific_models).float().to(param_dataset_ids.device)
    elif param_mode == 1: # uniform weights = > 1/num_dataset_specific_models for all dataset specific models
        return torch.full((param_dataset_ids.shape[0], param_num_dataset_specific_models), 1.0 / param_num_dataset_specific_models, device=param_dataset_ids.device, dtype=torch.float32)
    elif param_mode == 2: # biased weights = > 0.5 for the dataset specific model corresponding to the dataset and 0.25 for the others
        weights = torch.full((param_dataset_ids.shape[0], param_num_dataset_specific_models), 0.25, device=param_dataset_ids.device, dtype=torch.float32)
        return weights.scatter_(1, param_dataset_ids.view(-1, 1), 0.5)

# Function that saves a checkpoint for resuming training later if needed
def save_resume_checkpoint(param_model, param_epoch, param_optimizer, param_scheduler, param_best_validation_metric, param_num_epochs_no_improvement, param_path=RESUME_CHECKPOINT_PATH):
    torch.save({
        'model_state': param_model.state_dict(),
        'epoch': param_epoch,
        'optimizer_state': param_optimizer.state_dict(),
        'scheduler_state': param_scheduler.state_dict(),
        'best_validation_metric': param_best_validation_metric,
        'num_epochs_no_improvement': param_num_epochs_no_improvement,
        'rng_state': {
            'python': random.getstate(),
            'numpy': np.random.get_state(),
            'torch': torch.get_rng_state().cpu(),
            'cuda': [state.cpu() for state in torch.cuda.get_rng_state_all()]
        }
    }, param_path)

# Function that loads a checkpoint and restores the model, optimizer, scheduler states, as well as RNG states for reproducibility
def load_resume_checkpoint(param_model, param_optimizer, param_scheduler, param_path=RESUME_CHECKPOINT_PATH, param_device=DEVICE):
    checkpoint = torch.load(param_path, map_location=param_device, weights_only=False)
    param_model.load_state_dict(checkpoint['model_state'])
    param_optimizer.load_state_dict(checkpoint['optimizer_state'])
    param_scheduler.load_state_dict(checkpoint['scheduler_state'])
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

class BalancedBatchSampler(Sampler):
    '''
    For batch_size=16 and 3 datasets:
        batch 1 -> 6, 5, 5 samples from dataset 1, 2, 3
        batch 2 -> 5, 6, 5 samples from dataset 1, 2, 3
        batch 3 -> 5, 5, 6 samples from dataset 1, 2, 3
        and so on
    '''
    def __init__(self, param_dataset_ids, param_batch_size, param_shuffle=True, param_allow_incomplete_last_batch=False):
        super().__init__()
        self.dataset_ids = np.array(param_dataset_ids, dtype=np.int64)
        self.batch_size = param_batch_size
        self.shuffle = param_shuffle
        self.allow_incomplete_last_batch = param_allow_incomplete_last_batch
        self.epoch = 0

        self.unique_dataset_ids = sorted(np.unique(self.dataset_ids).tolist())
        self.indices_per_dataset = {dataset_id: np.where(self.dataset_ids == dataset_id)[0].tolist() for dataset_id in self.unique_dataset_ids}
        
        self.base_count = self.batch_size // len(self.indices_per_dataset)
        self.remainder = self.batch_size % len(self.indices_per_dataset)
        self.num_batches = int(np.ceil(len(self.dataset_ids) / self.batch_size))

    def set_epoch(self, param_epoch):
        self.epoch = param_epoch

    def __len__(self):
        return self.num_batches
    
    def __iter__(self):
        if self.shuffle:
            random_generator = random.Random(SEED + self.epoch)

        shuffled_indices, current_indices = {}, {}
        for dataset_id in self.unique_dataset_ids:
            dataset_indices = self.indices_per_dataset[dataset_id].copy()
            
            if self.shuffle:
                random_generator.shuffle(dataset_indices)
            
            shuffled_indices[dataset_id] = dataset_indices
            current_indices[dataset_id] = 0

        for batch_index in range(self.num_batches):
            sample_counts_per_dataset = {dataset_id: self.base_count for dataset_id in self.unique_dataset_ids}
            
            for extra_index in range(self.remainder):
                dataset_id = self.unique_dataset_ids[(batch_index + extra_index + self.epoch) % len(self.unique_dataset_ids)]
                sample_counts_per_dataset[dataset_id] += 1

            batch = []

            for dataset_id in self.unique_dataset_ids:
                num_required_samples = sample_counts_per_dataset[dataset_id]
                dataset_indices = shuffled_indices[dataset_id]
            
                if self.allow_incomplete_last_batch:
                    start_index = current_indices[dataset_id]
                    num_remaining_samples = len(dataset_indices) - start_index

                    num_samples_to_take = min(num_required_samples, num_remaining_samples)
                    if num_samples_to_take > 0:
                        end_index = start_index + num_samples_to_take
                        batch.extend(dataset_indices[start_index:end_index])
                        current_indices[dataset_id] = end_index
                else:
                    selected_samples = []
                    while len(selected_samples) < num_required_samples:
                        start_index = current_indices[dataset_id]
                        num_remaining_samples = len(dataset_indices) - start_index
                        needed_samples = num_required_samples - len(selected_samples)

                        if num_remaining_samples >= needed_samples:
                            end_index = start_index + needed_samples
                            selected_samples.extend(dataset_indices[start_index:end_index])
                            current_indices[dataset_id] = end_index
                        else:
                            if num_remaining_samples > 0:
                                selected_samples.extend(dataset_indices[start_index:])

                            if self.shuffle:
                                random_generator.shuffle(dataset_indices)

                            current_indices[dataset_id] = 0

                    batch.extend(selected_samples)

            if len(batch) == 0:
                break

            if self.shuffle:
                random_generator.shuffle(batch)
            yield batch

# Segmentation Dataset class for loading images and masks
class SegmentationDataset(Dataset):
    def __init__(self, param_images_path, param_masks_path, param_dataset_ids, param_size, param_transform=None):
        super().__init__()
        self.images_path = param_images_path
        self.masks_path = param_masks_path
        self.dataset_ids = param_dataset_ids
        self.num_samples = len(param_images_path)
        self.size = param_size
        self.transform = param_transform

    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, param_index):
        image = cv.imread(self.images_path[param_index], cv.IMREAD_COLOR)
        mask = cv.imread(self.masks_path[param_index], cv.IMREAD_GRAYSCALE)
        dataset_id = self.dataset_ids[param_index]

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations['image']
            mask = augmentations['mask']

        image = cv.resize(image, self.size, interpolation=cv.INTER_LINEAR)
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        image.div_(255.0)

        mask = cv.resize(mask, self.size, interpolation=cv.INTER_NEAREST)
        mask = (mask > 127).astype(np.float32)
        mask = torch.from_numpy(mask).unsqueeze(0)

        return image, mask, dataset_id


# RESNET BACKBONE
model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        
        self.groups = groups
        self.base_width = width_per_group
        
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        
        if dilate:
            self.dilation *= stride
            stride = 1
        
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width, dilation=self.dilation, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model

def resnet50(pretrained=True, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


# TResUnet model
class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.relu = nn.ReLU()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c)
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_c)
        )

    def forward(self, inputs):
        x1 = self.conv(inputs)
        x2 = self.shortcut(inputs)
        x = self.relu(x1 + x2)
        return x

class EncoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.r1 = ResidualBlock(in_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, inputs):
        x = self.r1(inputs)
        p = self.pool(x)
        return x, p

class BottleneckTResUnet(nn.Module):
    def __init__(self, in_c, out_c, dim, num_layers=2):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )

        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=8, batch_first=True)
        self.tblock = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv1(x)
        b, c, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.tblock(x)
        x = x.transpose(1, 2).reshape(b, c, h, w)
        x = self.conv2(x)
        return x

class DilatedConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.c1 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )

        self.c2 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=3, dilation=3),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )

        self.c3 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=6, dilation=6),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )

        self.c4 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=9, dilation=9),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )

        self.c5 = nn.Sequential(
            nn.Conv2d(out_c*4, out_c, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )

    def forward(self, inputs):
        x1 = self.c1(inputs)
        x2 = self.c2(inputs)
        x3 = self.c3(inputs)
        x4 = self.c4(inputs)
        x = torch.cat([x1, x2, x3, x4], dim=1)
        x = self.c5(x)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.r1 = ResidualBlock(in_c[0]+in_c[1], out_c)
        self.r2 = ResidualBlock(out_c, out_c)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], dim=1)
        x = self.r1(x)
        x = self.r2(x)
        return x

class TResUnet(nn.Module):
    def __init__(self):
        super().__init__()

        ''' ResNet50 '''
        backbone = resnet50()
        self.layer0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.layer1 = nn.Sequential(backbone.maxpool, backbone.layer1)
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3

        ''' Bridge blocks '''
        self.b1 = BottleneckTResUnet(1024, 256, 256, num_layers=2)
        self.b2 = DilatedConv(1024, 256)

        ''' Decoder '''
        self.d1 = DecoderBlock([512, 512], 256)
        self.d2 = DecoderBlock([256, 256], 128)
        self.d3 = DecoderBlock([128, 64], 64)
        self.d4 = DecoderBlock([64, 3], 32)

        self.output = nn.Conv2d(32, 1, kernel_size=1)

    def encode(self, x):
        s1 = self.layer0(x)
        s2 = self.layer1(s1)
        s3 = self.layer2(s2)
        s4 = self.layer3(s3)

        b1 = self.b1(s4)
        b2 = self.b2(s4)
        b3 = torch.cat([b1, b2], dim=1)

        return s1, s2, s3, b3

    def forward(self, x):
        s1 = self.layer0(x)    ## [-1, 64, h/2, w/2]
        s2 = self.layer1(s1)    ## [-1, 256, h/4, w/4]
        s3 = self.layer2(s2)    ## [-1, 512, h/8, w/8]
        s4 = self.layer3(s3)    ## [-1, 1024, h/16, w/16]

        b1 = self.b1(s4)
        b2 = self.b2(s4)
        b3 = torch.cat([b1, b2], dim=1)

        d1 = self.d1(b3, s3)
        d2 = self.d2(d1, s2)
        d3 = self.d3(d2, s1)
        d4 = self.d4(d3, x)

        return self.output(d4)
    
class ConvolveResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # Set out_channels to be one-third of in_channels
        self.out_channels = in_channels // 3

        self.conv1 = nn.Conv2d(in_channels, self.out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(self.out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(self.out_channels)

        # For matching dimensions in case the input and output channels are different
        self.shortcut = nn.Conv2d(in_channels, self.out_channels, kernel_size=1, stride=1,
                                   padding=0) if in_channels != self.out_channels else None

    def forward(self, x):
        # Apply the first convolutional layer followed by ReLU
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # Apply the second convolutional layer
        out = self.conv2(out)
        out = self.bn2(out)

        # Add the shortcut connection
        if self.shortcut is not None:
            x = self.shortcut(x)

        out += x  # Residual connection
        out = self.relu(out)  # Final activation

        return out
    
class CrossAttentionBlock(nn.Module):
    def __init__(self, in_channels, num_heads=8):
        super().__init__()
        self.query = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.num_heads = num_heads
        self.scale = (in_channels // num_heads) ** -0.5
        self.alpha = nn.Parameter(torch.ones(1))  # Learnable scaling factor

        # Additional layers for enhancing the model
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=1)
        )
        # Adjust normalization to handle channel dimension correctly
        self.norm1 = nn.LayerNorm(in_channels)  # This needs to be applied differently
        self.norm2 = nn.LayerNorm(in_channels)

    def forward(self, x1, x2):
        b, c, h, w = x1.shape

        # Calculate queries, keys, and values for both inputs
        q1 = self.query(x1).reshape(b, self.num_heads, c // self.num_heads, h * w)
        k2 = self.key(x2).reshape(b, self.num_heads, c // self.num_heads, h * w)
        v2 = self.value(x2).reshape(b, self.num_heads, c // self.num_heads, h * w)

        # Cross-attention between x1 and x2
        attn_weights = torch.einsum("bhqd,bhkd->bhqk", q1, k2) * self.scale
        attn_weights = attn_weights.softmax(dim=-1)

        # Apply attention weights to the values of x2
        attn_output = torch.einsum("bhqk,bhvd->bhqd", attn_weights, v2).reshape(b, c, h, w)

        # Apply normalization and a feed-forward network to enhance the attention output
        # Apply layer norm on the last dimension (channels)
        attn_output = attn_output.permute(0, 2, 3, 1).reshape(-1, c)  # Flatten for LayerNorm
        attn_output = self.norm1(attn_output)  # Normalizing the flattened tensor
        attn_output = attn_output.view(b, h, w, c).permute(0, 3, 1, 2)  # Reshape back
        attn_output = attn_output + x1  # Add the original input

        attn_output = self.fc(attn_output)
        attn_output = attn_output.permute(0, 2, 3, 1).reshape(-1, c)  # Flatten for LayerNorm
        attn_output = self.norm2(attn_output)  # Normalize again
        attn_output = attn_output.view(b, h, w, c).permute(0, 3, 1, 2)  # Reshape back

        # Dynamic weighted aggregation with residual
        return x1 + self.alpha * attn_output  # Learnable weight to adjust influence

class TResUnetGenericModel(nn.Module):
    def __init__(self, dataset_specific_paths=None, device=DEVICE):
        super().__init__()

        # Load pre-trained dataset-specific models
        self.dataset_specific_1 = TResUnet()
        self.dataset_specific_2 = TResUnet()
        self.dataset_specific_3 = TResUnet()

        if dataset_specific_paths is not None:
            assert len(dataset_specific_paths) == 3, 'Three dataset specific models paths must be provided.'
            self.dataset_specific_1.load_state_dict(torch.load(dataset_specific_paths[0], map_location=device), strict=False)
            self.dataset_specific_2.load_state_dict(torch.load(dataset_specific_paths[1], map_location=device), strict=False)
            self.dataset_specific_3.load_state_dict(torch.load(dataset_specific_paths[2], map_location=device), strict=False)

        # Cross-attention blocks for each encoder level
        self.cross_attn1 = CrossAttentionBlock(64)
        self.cross_attn2 = CrossAttentionBlock(256)
        self.cross_attn3 = CrossAttentionBlock(512)

        # Convolutional blocks for combined encoder outputs
        self.conv_1 = ConvolveResidualBlock(1536)
        self.conv_2 = ConvolveResidualBlock(1536)
        self.conv_3 = ConvolveResidualBlock(768)
        self.conv_4 = ConvolveResidualBlock(192)

        # Decoder blocks
        self.d1 = DecoderBlock([512, 512], 256)
        self.d2 = DecoderBlock([256, 256], 128)
        self.d3 = DecoderBlock([128, 64], 64)
        self.d4 = DecoderBlock([64, 3], 32)

        # Final output layer
        self.output = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        # Encode features from each dataset specific model
        ds1_s1, ds1_s2, ds1_s3, ds1_b = self.dataset_specific_1.encode(x)
        ds2_s1, ds2_s2, ds2_s3, ds2_b = self.dataset_specific_2.encode(x)
        ds3_s1, ds3_s2, ds3_s3, ds3_b = self.dataset_specific_3.encode(x)

        # Cross-attention on encoder outputs
        ds1_s1 = self.cross_attn1(ds1_s1, ds2_s1) + self.cross_attn1(ds1_s1, ds3_s1) + self.cross_attn1(ds2_s1, ds3_s1)
        ds1_s2 = self.cross_attn2(ds1_s2, ds2_s2) + self.cross_attn2(ds1_s2, ds3_s2) + self.cross_attn2(ds2_s2, ds3_s2)
        ds1_s3 = self.cross_attn3(ds1_s3, ds2_s3) + self.cross_attn3(ds1_s3, ds3_s3) + self.cross_attn3(ds2_s3, ds3_s3)

        # Concatenate the encoder outputs with cross-attention applied
        combined_s1 = torch.cat([ds1_s1, ds2_s1, ds3_s1], dim=1)
        combined_s2 = torch.cat([ds1_s2, ds2_s2, ds3_s2], dim=1)
        combined_s3 = torch.cat([ds1_s3, ds2_s3, ds3_s3], dim=1)

        # Concatenate bottleneck features from all dataset specific models
        combined_bottleneck = torch.cat((ds1_b, ds2_b, ds3_b), dim=1)

        # Convolution and decoder layers
        conv_bottleneck = self.conv_1(combined_bottleneck)
        conv_s3 = self.conv_2(combined_s3)
        conv_s2 = self.conv_3(combined_s2)
        conv_s1 = self.conv_4(combined_s1)

        # Decoder pass
        d1 = self.d1(conv_bottleneck, conv_s3)
        d2 = self.d2(d1, conv_s2)
        d3 = self.d3(d2, conv_s1)
        d4 = self.d4(d3, x)

        return self.output(d4)

class DiceBCELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets, smooth=1):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='mean')

        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2.0 * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        return bce_loss + dice_loss

def compute_feature_alignment_loss(param_cross_dataset_features, param_dataset_specific_features, param_weights):
    feature_alignment_loss = 0.0

    for encoder_block_index in range(len(param_cross_dataset_features)):
        cross_dataset_features = param_cross_dataset_features[encoder_block_index]

        for dataset_id in range(len(DATASETS_TO_IDS)):
            dataset_specific_feature = param_dataset_specific_features[dataset_id][encoder_block_index].detach()
            mse_difference_per_sample = F.mse_loss(cross_dataset_features, dataset_specific_feature, reduction='none').mean(dim=(1, 2, 3))
            feature_alignment_loss += (param_weights[:, dataset_id] * mse_difference_per_sample).mean()

    return feature_alignment_loss

        
def calculate_metrics(y_true, y_pred):
    y_true = y_true.detach().cpu().numpy()
    
    y_pred = torch.sigmoid(y_pred)
    y_pred = y_pred.detach().cpu().numpy()

    y_pred = y_pred > 0.5
    y_pred = y_pred.reshape(-1).astype(np.uint8)

    y_true = y_true > 0.5
    y_true = y_true.reshape(-1).astype(np.uint8)

    intersection = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum() - intersection
    score_precision = (intersection + 1e-15) / (y_pred.sum() + 1e-15)
    score_recall = (intersection + 1e-15) / (y_true.sum() + 1e-15)
    score_dice = (2.0 * intersection + 1e-15) / (y_true.sum() + y_pred.sum() + 1e-15)
    score_jaccard = (intersection + 1e-15) / (union + 1e-15)

    return [score_jaccard, score_dice, score_recall, score_precision]

# Function that performs a training step for the student model, including the feature alignment loss from the dataset specific models based on the specified weighting mode
def train_step(param_model, param_dataloader, param_optimizer, param_criterion, param_dataset_specific_models, param_weighting_mode, param_device):
    param_model.train()
    
    epoch_loss, epoch_jaccard, epoch_dice, epoch_recall, epoch_precision = 0.0, 0.0, 0.0, 0.0, 0.0

    for batched_images, batched_masks, batched_dataset_ids in param_dataloader:
        batched_images = batched_images.to(param_device, dtype=torch.float32, non_blocking=True)
        batched_masks = batched_masks.to(param_device, dtype=torch.float32, non_blocking=True)
        batched_dataset_ids = batched_dataset_ids.to(param_device, non_blocking=True, dtype=torch.long)

        param_optimizer.zero_grad()
        y_pred, cross_dataset_features = param_model(batched_images, return_feature_maps=True)
        dice_bce_loss = param_criterion(y_pred, batched_masks)

        # Pass the batch to each dataset specific model and collect the feature maps
        with torch.no_grad():
            # In this mode only the features from the dataset specific model corresponding to the dataset of each sample are used for feature alignment
            if param_weighting_mode == 0:
                dataset_specific_features = [torch.empty_like(feature) for feature in cross_dataset_features]
                # Iterate over the unique dataset ids in the batch and pass the corresponding samples through the appropriate dataset specific model to collect the dataset specific features
                for dataset_id in batched_dataset_ids.unique(sorted=False):
                    dataset_id = int(dataset_id.item())
                    indexes = (batched_dataset_ids == dataset_id).nonzero(as_tuple=True)[0]
                    dataset_specific_feature = param_dataset_specific_models[dataset_id].encode(batched_images[indexes])
                    for encoder_block_index in range(len(cross_dataset_features)):
                        dataset_specific_features[encoder_block_index][indexes] = dataset_specific_feature[encoder_block_index]

                # Compute the feature alignment loss using the collected dataset specific features and the student features
                feature_alignment_loss = 0.0
                for encoder_block_index in range(len(cross_dataset_features)):
                    mse_difference = F.mse_loss(cross_dataset_features[encoder_block_index], dataset_specific_features[encoder_block_index], reduction='none').mean(dim=(1, 2, 3))
                    feature_alignment_loss += mse_difference.mean()

            # Otherwise, the features are collected from all dataset specific models and weighted based on the specified mode to compute the feature alignment loss
            else:
                dataset_specific_features = [None] * len(param_dataset_specific_models)
                # Iterate over all dataset specific models and pass the entire batch through each model to collect the dataset specific features
                for dataset_id, dataset_specific_model in param_dataset_specific_models.items():
                    dataset_specific_features[dataset_id] = dataset_specific_model.encode(batched_images)

                # Determine the weights for each dataset specific model based on the specified mode and compute the feature alignment loss
                dataset_specific_models_weights = determine_weights_for_dataset_specific_models(batched_dataset_ids, param_weighting_mode, len(param_dataset_specific_models))
                feature_alignment_loss = compute_feature_alignment_loss(cross_dataset_features, dataset_specific_features, dataset_specific_models_weights)
    
        # Combine the segmentation loss and the feature alignment loss, perform backpropagation and update the model parameters
        total_loss = dice_bce_loss + feature_alignment_loss
        total_loss.backward()
        param_optimizer.step()

        epoch_loss += total_loss.item()

        # Calculate metrics
        batch_jaccard, batch_dice, batch_recall, batch_precision = [], [], [], []
        for yt, yp in zip(batched_masks, y_pred):
            score = calculate_metrics(yt, yp)
            batch_jaccard.append(score[0])
            batch_dice.append(score[1])
            batch_recall.append(score[2])
            batch_precision.append(score[3])

        epoch_jaccard += np.mean(batch_jaccard)
        epoch_dice += np.mean(batch_dice)
        epoch_recall += np.mean(batch_recall)
        epoch_precision += np.mean(batch_precision)

    epoch_loss /= len(param_dataloader)
    epoch_jaccard /= len(param_dataloader)
    epoch_dice /= len(param_dataloader)
    epoch_recall /= len(param_dataloader)
    epoch_precision /= len(param_dataloader)
    return epoch_loss, [epoch_jaccard, epoch_dice, epoch_recall, epoch_precision]

# Function that performs an evaluation step for the student model
def evaluate_step(param_model, param_dataloader, param_criterion, param_device):
    param_model.eval()

    epoch_loss, epoch_jaccard, epoch_dice, epoch_recall, epoch_precision = 0.0, 0.0, 0.0, 0.0, 0.0

    with torch.inference_mode():
        for batched_images, batched_masks, _ in param_dataloader:
            batched_images = batched_images.to(param_device, dtype=torch.float32, non_blocking=True)
            batched_masks = batched_masks.to(param_device, dtype=torch.float32, non_blocking=True)

            y_pred = param_model(batched_images)
            dice_bce_loss = param_criterion(y_pred, batched_masks)
            
            epoch_loss += dice_bce_loss.item()

            # Calculate metrics
            batch_jaccard, batch_dice, batch_recall, batch_precision = [], [], [], []
            for yt, yp in zip(batched_masks, y_pred):
                score = calculate_metrics(yt, yp)
                batch_jaccard.append(score[0])
                batch_dice.append(score[1])
                batch_recall.append(score[2])
                batch_precision.append(score[3])

            epoch_jaccard += np.mean(batch_jaccard)
            epoch_dice += np.mean(batch_dice)
            epoch_recall += np.mean(batch_recall)
            epoch_precision += np.mean(batch_precision)

    epoch_loss /= len(param_dataloader)
    epoch_jaccard /= len(param_dataloader)
    epoch_dice /= len(param_dataloader)
    epoch_recall /= len(param_dataloader)
    epoch_precision /= len(param_dataloader)
    return epoch_loss, [epoch_jaccard, epoch_dice, epoch_recall, epoch_precision]


if __name__ == '__main__':
    seed_all(SEED)

    if os.path.exists(LOG_PATH):
        print('Log file exists')
    else:
        train_log_file = open(LOG_PATH, 'w')
        train_log_file.write('\n')
        train_log_file.close()

    # Log the start time of training
    start_datetime = str(datetime.datetime.now())
    print_and_save(LOG_PATH, start_datetime)

    # Log hyperparameters
    hyperparameters_log_text = f'Image size: {HYPERPARAMETERS["image_size"]}\nBatch size: {HYPERPARAMETERS["batch_size"]}\nLR: {HYPERPARAMETERS["init_learning_rate"]}\n'
    hyperparameters_log_text += f'Epochs: {HYPERPARAMETERS["num_epochs"]}\nScheduler Patience: {HYPERPARAMETERS["scheduler_patience"]}\nEarly Stopping Patience: {HYPERPARAMETERS["early_stopping_patience"]}\n'
    hyperparameters_log_text += f'Weighting mode: {HYPERPARAMETERS["dsm_weight_mode"]}\n'
    print_and_save(LOG_PATH, hyperparameters_log_text)

    # Load the images and masks file names for training and validation
    (train_images_paths, train_masks_paths, train_dataset_ids), (validation_images_paths, validation_masks_paths, validation_dataset_ids) = load_data(DATASETS_PATHS)
    train_images_paths, train_masks_paths, train_dataset_ids = shuffling(train_images_paths, train_masks_paths, train_dataset_ids)
    dataset_log_text = f'Dataset Size:\nTrain: {len(train_images_paths)}\nValidation: {len(validation_images_paths)}\n'
    print_and_save(LOG_PATH, dataset_log_text)

    # Define data augmentation transforms using albumentations
    augmentation = A.Compose([
        A.Rotate(limit=35, p=0.3),
        A.HorizontalFlip(p=0.3),
        A.VerticalFlip(p=0.3),
        A.CoarseDropout(p=0.3, num_holes_range=(1, 10), hole_height_range=(1, 32), hole_width_range=(1, 32))
    ])

    # Create datasets for training and validation
    train_dataset = SegmentationDataset(train_images_paths, train_masks_paths, train_dataset_ids, HYPERPARAMETERS['image_size'], param_transform=augmentation)
    validation_dataset = SegmentationDataset(validation_images_paths, validation_masks_paths, validation_dataset_ids, HYPERPARAMETERS['image_size'])
    
    # Create balanced batch sampler for training dataset to ensure that each batch contains samples from each dataset according to the specified ratios
    train_sampler = BalancedBatchSampler(train_dataset.dataset_ids, HYPERPARAMETERS['batch_size'], param_shuffle=True, param_allow_incomplete_last_batch=False)
    validation_sampler = BalancedBatchSampler(validation_dataset.dataset_ids, HYPERPARAMETERS['batch_size'], param_shuffle=False, param_allow_incomplete_last_batch=True)

    # Create dataloaders for training and validation datasets
    train_dataloader = DataLoader(dataset=train_dataset, batch_sampler=train_sampler, num_workers=2, pin_memory=True, persistent_workers=True)
    validation_dataloader = DataLoader(dataset=validation_dataset, batch_sampler=validation_sampler, num_workers=2, pin_memory=True, persistent_workers=True)

    # Load dataset specific models checkpoints for each dataset
    dataset_specific_models = load_dataset_specific_models(DATASET_SPECIFIC_MODELS_CHECKPOINTS)

    # Create model, optimizer, scheduler, and criterion
    model = TResUnet().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=HYPERPARAMETERS['init_learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=HYPERPARAMETERS['scheduler_patience'])
    criterion = DiceBCELoss()

    start_epoch = 0
    best_validation_metric = 0.0
    num_epochs_no_improvement = 0

    # If resume checkpoint exists, load it
    if os.path.exists(RESUME_CHECKPOINT_PATH):
        start_epoch, best_validation_metric, num_epochs_no_improvement = load_resume_checkpoint(model, optimizer, scheduler, RESUME_CHECKPOINT_PATH, DEVICE)

    for epoch in range(start_epoch, HYPERPARAMETERS['num_epochs']):
        start_time = time.time()
        train_sampler.set_epoch(epoch)

        train_loss, train_metrics = train_step(model, train_dataloader, optimizer, criterion, dataset_specific_models, HYPERPARAMETERS['dsm_weight_mode'], DEVICE)
        validation_loss, validation_metrics = evaluate_step(model, validation_dataloader, criterion, DEVICE)
        scheduler.step(validation_loss)

        if validation_metrics[1] > best_validation_metric:
            data_str = f'Valid F1 improved from {best_validation_metric:2.4f} to {validation_metrics[1]:2.4f}. Saving checkpoint: {CHECKPOINT_PATH}'
            print_and_save(LOG_PATH, data_str)

            best_validation_metric = validation_metrics[1]
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            num_epochs_no_improvement = 0
        else:
            num_epochs_no_improvement += 1

        end_time = time.time()
        epoch_duration_min = int((end_time - start_time) / 60)
        epoch_duration_sec = int((end_time - start_time) - (epoch_duration_min * 60))
        epoch_log_text = f'Epoch {epoch + 1} | Epoch Time: {epoch_duration_min}m {epoch_duration_sec}s\n'
        epoch_log_text += f'\tTrain Loss: {train_loss:.4f} - Jaccard: {train_metrics[0]:.4f} - Dice (F1): {train_metrics[1]:.4f} - Recall: {train_metrics[2]:.4f} - Precision: {train_metrics[3]:.4f}\n'
        epoch_log_text += f'\tValidation Loss: {validation_loss:.4f} - Jaccard: {validation_metrics[0]:.4f} - Dice (F1): {validation_metrics[1]:.4f} - Recall: {validation_metrics[2]:.4f} - Precision: {validation_metrics[3]:.4f}\n'
        print_and_save(LOG_PATH, epoch_log_text)

        if num_epochs_no_improvement == HYPERPARAMETERS['early_stopping_patience']:
            print_and_save(LOG_PATH, f'Early stopping triggered after {epoch+1} epochs.')
            break

        save_resume_checkpoint(model, epoch, optimizer, scheduler, best_validation_metric, num_epochs_no_improvement, RESUME_CHECKPOINT_PATH)