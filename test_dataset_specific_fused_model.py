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
    'batch_size': 16
}

# Dictionary that maps dataset names to an id
DATASETS_TO_IDS = {'isles': 0, 'bmshare': 1, 'brats': 2}

# Constant for the root path for all necessary files
ROOT_PATH = '/root/Disertation'

# Constants for dataset paths
DATASETS_ROOT_PATH = f'{ROOT_PATH}/datasets'
DATASETS_PATHS = {dataset_name: os.path.join(DATASETS_ROOT_PATH, dataset_name) for dataset_name in DATASETS_TO_IDS.keys()}

# Constant for dataset specific models checkpoint paths and a mapping from dataset names to paths
DATASET_SPECIFIC_MODELS_ROOT_PATH = f'{ROOT_PATH}/files/dataset_specific'
DATASET_SPECIFIC_MODELS_CHECKPOINTS = {dataset_name: os.path.join(DATASET_SPECIFIC_MODELS_ROOT_PATH, dataset_name, f'dataset_specific_model_{dataset_name}.pth') for dataset_name in DATASETS_TO_IDS.keys()}

# Constants for model checkpoint path and log path
MODELS_AND_LOG_ROOT_PATH = f'{ROOT_PATH}/files/fused_dataset_specific/not_weighted'
CHECKPOINT_PATH = f'{MODELS_AND_LOG_ROOT_PATH}/fused_dataset_specific_model.pth'
LOG_PATH = f'{MODELS_AND_LOG_ROOT_PATH}/test_log_fused_dataset_specific.txt'

# Function that sets constant seed for reproducibility
def seed_all(param_seed=SEED):
    random.seed(param_seed)
    os.environ['PYTHONHASHSEED'] = str(param_seed)
    np.random.seed(param_seed)
    torch.manual_seed(param_seed)
    torch.cuda.manual_seed(param_seed)
    torch.cuda.manual_seed_all(param_seed)
    torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False

# Function that prints a message and also saves it to a specified file
def print_and_save(param_file_path, param_text):
    print(param_text)
    with open(param_file_path, 'a') as file:
        file.write(param_text)
        file.write('\n')

# Function that loads all file names for images and masks in the dataset
def load_split_filenames(param_dataset_path, param_split_file):
    with open(param_split_file, 'r') as f:
        file_names = [line.strip() for line in f if line.strip()]
    
    images = [os.path.join(param_dataset_path, 'images', name) for name in file_names]
    masks = [os.path.join(param_dataset_path, 'masks', name) for name in file_names]
    return images, masks

# Function that loads test data from specified dataset path
def load_test_data(param_dataset_path):
    test_split_file = os.path.join(param_dataset_path, 'test.txt')
    test_images_path, test_masks_path = load_split_filenames(param_dataset_path, test_split_file)
    return (test_images_path, test_masks_path)

# Function that loads dataset specific models from specified checkpoints and returns a dictionary that maps dataset ids to the corresponding model
def load_dataset_specific_models(param_dataset_specific_models_checkpoints, param_device=DEVICE):
    dataset_specific_models = {}
    for dataset_name, checkpoint_path in param_dataset_specific_models_checkpoints.items():
        dataset_specific_model = TResUnet().to(param_device)
        dataset_specific_model.load_state_dict(torch.load(checkpoint_path, map_location=param_device), strict=False)
        dataset_specific_model.eval()

        # Freeze the parameters of each dataset specific model
        for param in dataset_specific_model.parameters():
            param.requires_grad = False

        dataset_specific_models[DATASETS_TO_IDS[dataset_name]] = dataset_specific_model
    return dataset_specific_models

# Determine the weights for each dataset specific model based on the specified mode and the dataset ids of the samples in the batch
def determine_weights_for_dataset_specific_models(param_dataset_ids, param_mode, param_num_dataset_specific_models=len(DATASETS_TO_IDS)):
    if param_mode == 0: # one hot encoding = > 1 for the dataset specific model corresponding to the dataset and 0 for the others
        return F.one_hot(param_dataset_ids, num_classes=param_num_dataset_specific_models).float().to(param_dataset_ids.device)
    elif param_mode == 1: # uniform weights = > 1/num_dataset_specific_models for all dataset specific models
        return torch.full((param_dataset_ids.shape[0], param_num_dataset_specific_models), 1.0 / param_num_dataset_specific_models, device=param_dataset_ids.device, dtype=torch.float32)
    elif param_mode == 2: # biased weights = > 0.5 for the dataset specific model corresponding to the dataset and 0.25 for the others
        weights = torch.full((param_dataset_ids.shape[0], param_num_dataset_specific_models), 0.25, device=param_dataset_ids.device, dtype=torch.float32)
        return weights.scatter_(1, param_dataset_ids.view(-1, 1), 0.5)

# Segmentation Dataset class for loading images and masks
class SegmentationDataset(Dataset):
    def __init__(self, param_images_path, param_masks_path, param_size, param_transform=None):
        super().__init__()
        self.images_path = param_images_path
        self.masks_path = param_masks_path
        self.num_samples = len(param_images_path)
        self.size = param_size
        self.transform = param_transform

    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, param_index):
        image = cv.imread(self.images_path[param_index], cv.IMREAD_COLOR)
        mask = cv.imread(self.masks_path[param_index], cv.IMREAD_GRAYSCALE)

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

        return image, mask
    

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

        return [s1, s2, s3, b3]

    def forward(self, x, return_features=False):
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

        y = self.output(d4)

        if return_features:
            return y, [s1, s2, s3, b3]
        return y
    
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
    
'''class CrossAttentionBlock(nn.Module):
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
        return x1 + self.alpha * attn_output  # Learnable weight to adjust influence'''

class TResUnetFusedDatasetSpecificModels(nn.Module):
    def __init__(self, dataset_specific_models):
        super().__init__()
        # Load the pretrained dataset specific models and the weighting mode for combining their features
        self.dataset_specific_1 = dataset_specific_models[0]
        self.dataset_specific_2 = dataset_specific_models[1]
        self.dataset_specific_3 = dataset_specific_models[2]

        # Cross-attention blocks for each encoder level
        #self.cross_attn1 = CrossAttentionBlock(64)
        #self.cross_attn2 = CrossAttentionBlock(256)
        #self.cross_attn3 = CrossAttentionBlock(512)

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

    def forward(self, x, weighting_mode=None, dataset_ids=None, return_features=False):
        # Encode features from each dataset specific model
        [ds1_s1, ds1_s2, ds1_s3, ds1_b] = self.dataset_specific_1.encode(x)
        [ds2_s1, ds2_s2, ds2_s3, ds2_b] = self.dataset_specific_2.encode(x)
        [ds3_s1, ds3_s2, ds3_s3, ds3_b] = self.dataset_specific_3.encode(x)

        # Cross-attention on encoder outputs
        #ds1_s1 = self.cross_attn1(ds1_s1, ds2_s1) + self.cross_attn1(ds1_s1, ds3_s1) + self.cross_attn1(ds2_s1, ds3_s1)
        #ds1_s2 = self.cross_attn2(ds1_s2, ds2_s2) + self.cross_attn2(ds1_s2, ds3_s2) + self.cross_attn2(ds2_s2, ds3_s2)
        #ds1_s3 = self.cross_attn3(ds1_s3, ds2_s3) + self.cross_attn3(ds1_s3, ds3_s3) + self.cross_attn3(ds2_s3, ds3_s3)

        if weighting_mode is not None:
            weights = determine_weights_for_dataset_specific_models(dataset_ids, weighting_mode)
            weights_1 = weights[:, 0].view(-1, 1, 1, 1)
            weights_2 = weights[:, 1].view(-1, 1, 1, 1)
            weights_3 = weights[:, 2].view(-1, 1, 1, 1)

            # Concatenate the encoder outputs with cross-attention applied
            combined_s1 = torch.cat([weights_1 * ds1_s1, weights_2 * ds2_s1, weights_3 * ds3_s1], dim=1)
            combined_s2 = torch.cat([weights_1 * ds1_s2, weights_2 * ds2_s2, weights_3 * ds3_s2], dim=1)
            combined_s3 = torch.cat([weights_1 * ds1_s3, weights_2 * ds2_s3, weights_3 * ds3_s3], dim=1)

            # Concatenate bottleneck features from all dataset specific models
            combined_bottleneck = torch.cat((weights_1 * ds1_b, weights_2 * ds2_b, weights_3 * ds3_b), dim=1)
        else:
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

        y = self.output(d4)

        if return_features:
            return y, [conv_s1, conv_s2, conv_s3, conv_bottleneck]
        else:
            return y

class DiceBCELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets, smooth=1):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='mean')

        inputs = torch.sigmoid(inputs)
        #inputs = inputs.view(-1)
        #targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2.0 * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        return bce_loss + dice_loss
    

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

# Function that performs an evaluation step for the student model
def evaluate_step(param_model, param_dataloader, param_criterion, param_device):
    param_model.eval()

    epoch_loss, epoch_jaccard, epoch_dice, epoch_recall, epoch_precision = 0.0, 0.0, 0.0, 0.0, 0.0

    with torch.inference_mode():
        for batched_images, batched_masks in param_dataloader:
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

    for dataset_name, dataset_path in DATASETS_PATHS.items():
        dataset_log_text = f'{dataset_name} dataset path: {dataset_path}'
        print_and_save(LOG_PATH, dataset_log_text)

        # Load the images and masks file names for the test split
        (test_images_paths, test_masks_paths) = load_test_data(dataset_path)
        dataset_log_text = f'Test set size: {len(test_images_paths)}\n'
        print_and_save(LOG_PATH, dataset_log_text)

        # Create dataset for test
        test_dataset = SegmentationDataset(test_images_paths, test_masks_paths, HYPERPARAMETERS['image_size'])

        # Create dataloaders
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=HYPERPARAMETERS['batch_size'], shuffle=False, num_workers=0, pin_memory=True)

        # Load dataset specific models checkpoints for each dataset
        dataset_specific_models = load_dataset_specific_models(DATASET_SPECIFIC_MODELS_CHECKPOINTS)

        # Create model, optimizer, scheduler, and criterion
        model = TResUnetFusedDatasetSpecificModels(dataset_specific_models).to(DEVICE)
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE), strict=False)

        test_loss, test_metrics = evaluate_step(model, test_dataloader, DiceBCELoss(), DEVICE)
        test_log_text = f'Test Loss: {test_loss:.4f} - Jaccard: {test_metrics[0]:.4f} - Dice (F1): {test_metrics[1]:.4f} - Recall: {test_metrics[2]:.4f} - Precision: {test_metrics[3]:.4f}\n\n'
        print_and_save(LOG_PATH, test_log_text)