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
from sklearn.metrics import accuracy_score
import albumentations as A

cv.setNumThreads(0)
# Set a fixed seed value
SEED = 42
# Set the device to cuda
DEVICE = torch.device('cuda')

# Constant for hyperparameters (moved here for claity and easy modification)
HYPERPARAMETERS = {
    'image_size': (256, 256),
    'batch_size': 8,
    'num_epochs': 300,
    'init_learning_rate': 0.0001,
    'scheduler_patience': 5,
    'early_stopping_patience': 35,
    'teacher_weighting_mode': 2, # 0 for one hot encoding, 1 for uniform weights, 2 for [0.5, 0.25, 0.25] weights
    'batch_ratios': {0: 2, 1: 2, 2: 4} # for balanced batch sampler, the number of samples from each dataset in a batch
}

# Constant with dataset names
DATASET_NAMES = ['isles', 'bmshare', 'brats']
# Dictionary that maps dataset names to an id
DATASETS_TO_IDS = {'isles': 0, 'bmshare': 1, 'brats': 2}

# Constants for dataset paths
DATASETS_ROOT_PATH = '/home/dragos/disertation/datasets'
DATASETS_PATHS = {dataset_name: os.path.join(DATASETS_ROOT_PATH, dataset_name) for dataset_name in DATASET_NAMES}
# Constants for student model checkpoint path and log path
os.makedirs('/home/dragos/disertation/files/student/weighted', exist_ok=True)
CHECKPOINT_PATH = '/home/dragos/disertation/files/student/weighted/best_student_model.pth'
LOG_PATH = '/home/dragos/disertation/files/student/weighted/train_log.txt'
# Constant for teacher models checkpoint paths and a mapping from dataset names to paths
TEACHER_MODELS_ROOT_PATH = '/home/dragos/disertation/files'
TEACHER_MODELS_CHECKPOINTS = {dataset_name: os.path.join(TEACHER_MODELS_ROOT_PATH, dataset_name, f'teacher_model_{dataset_name}.pth') for dataset_name in DATASET_NAMES}

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
def load_file_names(param_dataset_path, param_split_file):
    file_names = open(param_split_file, 'r').read().split('\n')[:-1]
    images = [os.path.join(param_dataset_path, 'images', name) for name in file_names]
    masks = [os.path.join(param_dataset_path, 'masks', name) for name in file_names]
    return images, masks

# Function that loads training and validation data from specified dataset path
def load_data(param_dataset_paths):
    all_train_images, all_train_masks, all_train_dataset_ids = [], [], []
    all_validation_images, all_validation_masks, all_validation_dataset_ids = [], [], []

    for dataset_name, dataset_path in param_dataset_paths.items():
        train_images, train_masks = load_file_names(dataset_path, os.path.join(dataset_path, 'train.txt'))
        validation_images, validation_masks = load_file_names(dataset_path, os.path.join(dataset_path, 'val.txt'))

        dataset_id = DATASETS_TO_IDS[dataset_name]
        all_train_images.extend(train_images)
        all_train_masks.extend(train_masks)
        all_train_dataset_ids.extend([dataset_id] * len(train_images))

        all_validation_images.extend(validation_images)
        all_validation_masks.extend(validation_masks)
        all_validation_dataset_ids.extend([dataset_id] * len(validation_images))

    return (all_train_images, all_train_masks, all_train_dataset_ids), (all_validation_images, all_validation_masks, all_validation_dataset_ids)

def shuffling(param_images_path, param_masks_path, param_dataset_ids):
    param_images_path, param_masks_path, param_dataset_ids = shuffle(param_images_path, param_masks_path, param_dataset_ids, random_state=SEED)
    return param_images_path, param_masks_path, param_dataset_ids

# Function that loads teacher models from specified checkpoints and returns a dictionary that maps dataset ids to the corresponding teacher model
def load_teacher_models(param_teacher_models_checkpoints):
    teacher_models = {}
    for dataset_name, checkpoint_path in param_teacher_models_checkpoints.items():
        teacher_model = TResUnet().to(DEVICE)
        teacher_model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
        teacher_model.eval()

        # Freeze the parameters of each teacher model
        for param in teacher_model.parameters():
            param.requires_grad = False

        teacher_models[DATASETS_TO_IDS[dataset_name]] = teacher_model
    return teacher_models

# Determine the weights for each teacher based on the specified mode and the dataset ids of the samples in the batch
def determine_teachers_weights(param_dataset_ids, param_mode=0, param_num_teachers=len(DATASET_NAMES)):
    if param_mode == 0: # one hot encoding = > 1 for the teacher corresponding to the dataset and 0 for the others
        return F.one_hot(param_dataset_ids, num_classes=param_num_teachers).float().to(DEVICE)
    elif param_mode == 1: # uniform weights = > 1/num_teachers for all teachers
        return torch.full((param_dataset_ids.shape[0], param_num_teachers), 1.0 / param_num_teachers, device=DEVICE, dtype=torch.float32)
    elif param_mode == 2: # custom weights = > 0.5 for the teacher corresponding to the dataset and 0.25 for the others
        weights = torch.full((param_dataset_ids.shape[0], param_num_teachers), 0.25, device=DEVICE, dtype=torch.float32)
        return weights.scatter_(1, param_dataset_ids.view(-1, 1), 0.5)

class BalancedBatchSampler(Sampler):
    def __init__(self, param_dataset_ids, param_batch_ratios):
        super().__init__()

        self.dataset_ids = np.array(param_dataset_ids, dtype=np.int64)
        self.batch_ratios = param_batch_ratios
        self.epoch = 0

        self.ids_indices = {dataset_id: np.where(self.dataset_ids == dataset_id)[0] for dataset_id in self.batch_ratios.keys()}

        batches_per_dataset = []
        for dataset_id in self.batch_ratios.keys():
            num_samples = len(self.ids_indices[dataset_id])
            num_batches = int(np.ceil(num_samples / self.batch_ratios[dataset_id]))
            batches_per_dataset.append(num_batches)

        self.num_batches = max(batches_per_dataset)

    def set_epoch(self, param_epoch):
        self.epoch = param_epoch

    def __len__(self):
        return self.num_batches
    
    def __iter__(self):
        random_generator = random.Random(SEED + self.epoch)
        
        shuffled_indices, current_indices = {}, {}
        for dataset_id in self.batch_ratios.keys():
            dataset_indices = self.ids_indices[dataset_id].tolist()
            random_generator.shuffle(dataset_indices)
            shuffled_indices[dataset_id] = dataset_indices
            current_indices[dataset_id] = 0

        for _ in range(self.num_batches):
            batch =[]

            for dataset_id, num_samples_per_batch in self.batch_ratios.items():
                dataset_indices = shuffled_indices[dataset_id]
                start_idx = current_indices[dataset_id]
                end_idx = start_idx + num_samples_per_batch

                if end_idx > len(dataset_indices):
                    random_generator.shuffle(dataset_indices)
                    start_idx = 0
                    end_idx = num_samples_per_batch

                batch.extend(dataset_indices[start_idx:end_idx])
                current_indices[dataset_id] = end_idx

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

        image = cv.resize(image, self.size)
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        mask = cv.resize(mask, self.size)
        mask = torch.from_numpy(mask).unsqueeze(0).float() / 255.0

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
def save_feats_mean(x):
    _, _, h, _ = x.shape
    if h == 256:
        with torch.no_grad():
            x = x.detach().cpu().numpy()
            x = np.transpose(x[0], (1, 2, 0))
            x = np.mean(x, axis=-1)
            x = x/np.max(x)
            x = x * 255.0
            x = x.astype(np.uint8)
            x = cv.applyColorMap(x, cv.COLORMAP_JET)
            x = np.array(x, dtype=np.uint8)
            return x

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
        s0 = x
        s1 = self.layer0(s0)
        s2 = self.layer1(s1)
        s3 = self.layer2(s2)
        #s4 = self.layer3(s3)

        return [s1, s2, s3]

    def forward(self, x, return_feature_maps=False, heatmap=None):
        s0 = x
        s1 = self.layer0(s0)    ## [-1, 64, h/2, w/2]
        s2 = self.layer1(s1)    ## [-1, 256, h/4, w/4]
        s3 = self.layer2(s2)    ## [-1, 512, h/8, w/8]
        s4 = self.layer3(s3)    ## [-1, 1024, h/16, w/16]

        b1 = self.b1(s4)
        b2 = self.b2(s4)
        b3 = torch.cat([b1, b2], dim=1)

        d1 = self.d1(b3, s3)
        d2 = self.d2(d1, s2)
        d3 = self.d3(d2, s1)
        d4 = self.d4(d3, s0)

        y = self.output(d4)

        if return_feature_maps:
            feature_maps = [s1, s2, s3]

        if heatmap is not None:
            hmap = save_feats_mean(d4)
            if return_feature_maps:
                return hmap, y, feature_maps
            return hmap, y
        else:
            if return_feature_maps:
                return y, feature_maps
            return y

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()
        self.BCE = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets, smooth=1):
        bce_loss = self.BCE(inputs, targets)
        inputs = torch.sigmoid(inputs)

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1.0 - (2.*intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        return bce_loss + dice_loss

def compute_feature_distillation_loss(param_student_features, param_teachers_features, param_teachers_weights):
    distillation_loss = 0.0

    for encoder_block_index in range(len(param_student_features)):
        student_feature = param_student_features[encoder_block_index]

        for dataset_id in range(len(DATASET_NAMES)):
            teacher_feature = param_teachers_features[dataset_id][encoder_block_index].detach()
            mse_difference_per_sample = F.mse_loss(student_feature, teacher_feature, reduction='none').mean(dim=(1, 2, 3))
            distillation_loss += (param_teachers_weights[:, dataset_id] * mse_difference_per_sample).mean()

    return distillation_loss

def precision(y_true, y_pred):
    intersection = (y_true * y_pred).sum()
    return (intersection + 1e-15) / (y_pred.sum() + 1e-15)

def recall(y_true, y_pred):
    intersection = (y_true * y_pred).sum()
    return (intersection + 1e-15) / (y_true.sum() + 1e-15)

def F2(y_true, y_pred, beta=2):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return (1 + beta**2.) * (p*r) / float(beta**2 * p + r + 1e-15)

def dice_score(y_true, y_pred):
    return (2 * (y_true * y_pred).sum() + 1e-15) / (y_true.sum() + y_pred.sum() + 1e-15)

def jac_score(y_true, y_pred):
    intersection = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum() - intersection
    return (intersection + 1e-15) / (union + 1e-15)
        
def calculate_metrics(y_true, y_pred):
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()

    y_pred = y_pred > 0.5
    y_pred = y_pred.reshape(-1)
    y_pred = y_pred.astype(np.uint8)

    y_true = y_true > 0.5
    y_true = y_true.reshape(-1)
    y_true = y_true.astype(np.uint8)

    # Compute the scores for each metric
    score_jaccard = jac_score(y_true, y_pred)
    score_dice = dice_score(y_true, y_pred)
    score_recall = recall(y_true, y_pred)
    score_precision = precision(y_true, y_pred)
    #score_fbeta = F2(y_true, y_pred)
    #score_acc = accuracy_score(y_true, y_pred)

    return [score_jaccard, score_dice, score_recall, score_precision]#, score_acc, score_fbeta]

# Function that performs a training step for the student model, including the distillation loss from the teacher models based on the specified weighting mode
def train_step(param_model, param_dataloader, param_optimizer, param_criterion, param_teacher_models, param_teacher_weight_mode, param_device):
    param_model.train()
    
    epoch_loss = 0.0
    epoch_jaccard = 0.0
    epoch_dice = 0.0
    epoch_recall = 0.0
    epoch_precision = 0.0

    for batched_images, batched_masks, batched_dataset_ids in param_dataloader:
        batched_images = batched_images.to(param_device, non_blocking=True)
        batched_masks = batched_masks.to(param_device, non_blocking=True)
        batched_dataset_ids = batched_dataset_ids.to(param_device, non_blocking=True)

        param_optimizer.zero_grad(set_to_none=True)
        y_pred, student_features = param_model(batched_images, return_feature_maps=True)
        dice_bce_loss = param_criterion(y_pred, batched_masks)

        # Pass the batch to each teacher model and collect the feature maps
        with torch.no_grad():
            # In this mode only the features from the teacher corresponding to the dataset of each sample are used for distillation
            if param_teacher_weight_mode == 0:
                teacher_features_by_dataset_id = [torch.empty_like(sf) for sf in student_features]
                # Iterate over the unique dataset ids in the batch and pass the corresponding samples through the appropriate teacher model to collect the teacher features
                for dataset_id in batched_dataset_ids.unique(sorted=False):
                    dataset_id = int(dataset_id.item())
                    indexes = (batched_dataset_ids == dataset_id).nonzero(as_tuple=True)[0]
                    teacher_feats = param_teacher_models[dataset_id].encode(batched_images[indexes])
                    for encoder_block_index in range(len(student_features)):
                        teacher_features_by_dataset_id[encoder_block_index][indexes] = teacher_feats[encoder_block_index]

                # Compute the distillation loss using the collected teacher features and the student features
                distillation_loss = 0.0
                for encoder_block_index in range(len(student_features)):
                    distillation_loss += F.mse_loss(student_features[encoder_block_index], teacher_features_by_dataset_id[encoder_block_index], reduction='mean')
            # Otherwise, the features are collected from all teacher models and weighted based on the specified mode to compute the distillation loss
            else:
                teachers_features = [None] * len(param_teacher_models)
                for dataset_id, teacher_model in param_teacher_models.items():
                    teachers_features[dataset_id] = teacher_model.encode(batched_images)

                # Determine the weights for each teacher based on the specified mode and compute the distillation loss
                teachers_weights = determine_teachers_weights(batched_dataset_ids, param_teacher_weight_mode, len(param_teacher_models))
                distillation_loss = compute_feature_distillation_loss(student_features, teachers_features, teachers_weights)

        # Combine the segmentation loss and the distillation loss, perform backpropagation and update the model parameters
        total_loss = dice_bce_loss + distillation_loss
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

    epoch_loss = 0.0
    epoch_jaccard = 0.0
    epoch_dice = 0.0
    epoch_recall = 0.0
    epoch_precision = 0.0

    with torch.inference_mode():
        for batched_images, batched_masks, batched_dataset_ids in param_dataloader:
            batched_images = batched_images.to(param_device, non_blocking=True)
            batched_masks = batched_masks.to(param_device, non_blocking=True)
            batched_dataset_ids = batched_dataset_ids.to(param_device, non_blocking=True)

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

'''
def evaluate_step(param_model, param_dataloader, param_criterion, param_teacher_models, param_teacher_weight_mode, param_device):
    param_model.eval()

    epoch_loss = 0.0
    epoch_jaccard = 0.0
    epoch_dice = 0.0
    epoch_recall = 0.0
    epoch_precision = 0.0

    with torch.no_grad():
        for batched_images, batched_masks, batched_dataset_ids in param_dataloader:
            batched_images = batched_images.to(param_device, dtype=torch.float32)
            batched_masks = batched_masks.to(param_device, dtype=torch.float32)
            batched_dataset_ids = batched_dataset_ids.to(param_device, dtype=torch.int64)

            y_pred, student_features = param_model(batched_images, return_feature_maps=True)
            dice_bce_loss = param_criterion(y_pred, batched_masks)

            # Pass the batch to each teacher model and collect the feature maps
            teachers_features = [None] * len(param_teacher_models)
            with torch.inference_mode():
                for dataset_id, teacher_model in param_teacher_models.items():
                    _, teacher_features = teacher_model(batched_images, return_feature_maps=True)
                    teachers_features[dataset_id] = teacher_features

            # Determine the weights for each teacher based on the specified mode and compute the distillation loss
            teachers_weights = determine_teachers_weights(batched_dataset_ids, param_teacher_weight_mode, len(param_teacher_models))
            distillation_loss = compute_feature_distillation_loss(student_features, teachers_features, teachers_weights)

            total_loss = dice_bce_loss + distillation_loss
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
'''

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
    hyperparameters_log_text = f'Image size: {HYPERPARAMETERS["image_size"]}\nBatch size: {HYPERPARAMETERS["batch_size"]}\nLR: {HYPERPARAMETERS["init_learning_rate"]}\nEpochs: {HYPERPARAMETERS["num_epochs"]}\n'
    hyperparameters_log_text += f'Scheduler Patience: {HYPERPARAMETERS["scheduler_patience"]}\nEarly Stopping Patience: {HYPERPARAMETERS["early_stopping_patience"]}\nWeighting mode: {HYPERPARAMETERS["teacher_weighting_mode"]}\nBatch ratios: {HYPERPARAMETERS["batch_ratios"]}\n'
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
    train_sampler = BalancedBatchSampler(train_dataset.dataset_ids, HYPERPARAMETERS['batch_ratios'])

    # Create dataloaders
    train_dataloader = DataLoader(dataset=train_dataset, batch_sampler=train_sampler, num_workers=8, pin_memory=True, persistent_workers=True)
    validation_dataloader = DataLoader(dataset=validation_dataset, batch_size=HYPERPARAMETERS['batch_size'], shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True)

    # Load teacher models checkpoints for each dataset
    teacher_models = load_teacher_models(TEACHER_MODELS_CHECKPOINTS)

    # Create model, optimizer, scheduler, and criterion
    model = TResUnet().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=HYPERPARAMETERS['init_learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=HYPERPARAMETERS['scheduler_patience'])
    criterion = DiceBCELoss()

    best_validation_metric = 0.0
    num_epochs_no_improvement = 0

    for epoch in range(HYPERPARAMETERS['num_epochs']):
        start_time = time.time()
        train_sampler.set_epoch(epoch)

        train_loss, train_metrics = train_step(model, train_dataloader, optimizer, criterion, teacher_models, HYPERPARAMETERS['teacher_weighting_mode'], DEVICE)
        validation_loss, validation_metrics = evaluate_step(model, validation_dataloader, criterion, DEVICE)#teacher_models, HYPERPARAMETERS['teacher_weighting_mode'], DEVICE)
        scheduler.step(validation_loss)

        if validation_metrics[1] > best_validation_metric:
            data_str = f'Valid F1 improved from {best_validation_metric:2.4f} to {validation_metrics[1]:2.4f}. Saving checkpoint: {CHECKPOINT_PATH}'
            print_and_save(LOG_PATH, data_str)

            best_validation_metric = validation_metrics[1]
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            num_epochs_no_improvement = 0
        elif validation_metrics[1] < best_validation_metric:
            num_epochs_no_improvement += 1

        end_time = time.time()
        epoch_duration_min = int((end_time - start_time) / 60)
        epoch_duration_sec = int((end_time - start_time) - (epoch_duration_min * 60))
        epoch_log_text = f'Epoch {epoch+1} | Epoch Time: {epoch_duration_min}m {epoch_duration_sec}s\n'
        epoch_log_text += f'\tTrain Loss: {train_loss:.4f} - Jaccard: {train_metrics[0]:.4f} - Dice (F1): {train_metrics[1]:.4f} - Recall: {train_metrics[2]:.4f} - Precision: {train_metrics[3]:.4f}\n'
        epoch_log_text += f'\tValidation Loss: {validation_loss:.4f} - Jaccard: {validation_metrics[0]:.4f} - Dice (F1): {validation_metrics[1]:.4f} - Recall: {validation_metrics[2]:.4f} - Precision: {validation_metrics[3]:.4f}\n'
        print_and_save(LOG_PATH, epoch_log_text)

        if num_epochs_no_improvement == HYPERPARAMETERS['early_stopping_patience']:
            print_and_save(LOG_PATH, f'Early stopping triggered after {epoch+1} epochs.')
            break