import os
import random
import datetime
import time
import numpy as np
import cv2 as cv
from sklearn.utils import shuffle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
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
    'num_epochs': 150,
    'init_learning_rate': 0.0005,
    'scheduler_patience': 5,
    'early_stopping_patience': 25,
}

# Constants for dataset name and path
DATASET_NAME = 'isles' # 'bmshare', 'brats'
DATASET_PATH = f'/home/dragos/disertation/datasets/{DATASET_NAME}'
# Constant for model checkpoint path and log path
MODELS_AND_LOG_ROOT_PATH = f'/home/dragos/disertation/files/fine_tune/biased/s2_s3_features/{DATASET_NAME}'
os.makedirs(MODELS_AND_LOG_ROOT_PATH, exist_ok=True)
CHECKPOINT_PATH = f'{MODELS_AND_LOG_ROOT_PATH}/fine_tuned_{DATASET_NAME}.pth'
LOG_PATH = f'{MODELS_AND_LOG_ROOT_PATH}/train_log_{DATASET_NAME}.txt'
# Path to the pretrained model checkpoint
PRETRAINED_CHECKPOINT_PATH = '/home/dragos/disertation/files/cross_dataset/biased/s2_s3_features/cross_dataset_model.pth'

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

# Function that loads training and validation data from specified dataset path
def load_data(param_dataset_path):
    train_split_file = os.path.join(param_dataset_path, 'train.txt')
    validation_split_file = os.path.join(param_dataset_path, 'val.txt')

    train_images_path, train_masks_path = load_split_filenames(param_dataset_path, train_split_file)
    validation_images_path, validation_masks_path = load_split_filenames(param_dataset_path, validation_split_file)

    return (train_images_path, train_masks_path), (validation_images_path, validation_masks_path)

# Function that shuffles the images and masks
def shuffle_data(param_images_path, param_masks_path):
    param_images_path, param_masks_path = shuffle(param_images_path, param_masks_path, random_state=SEED)
    return param_images_path, param_masks_path

def set_bn_eval(param_module):
    if isinstance(param_module, nn.BatchNorm2d):
        param_module.eval()

def freeze_module(param_module):
    for param in param_module.parameters():
        param.requires_grad = False
    param_module.apply(set_bn_eval)

# Function that loads pretrained model and freezes the encoder layers for fine-tuning
def load_pretrained_model(param_pretrained_checkpoint_path, param_device=DEVICE, param_log_file_path=LOG_PATH):
    pretrained_model = TResUnet().to(param_device)
    pretrained_model.load_state_dict(torch.load(param_pretrained_checkpoint_path, map_location=param_device))
    
    # Freeze encoder layers
    for num_layer in range(4):
        layer = getattr(pretrained_model, f'layer{num_layer}')
        freeze_module(layer)

    # Freeze the bottleneck blocks
    #for num_block in range(1, 3):
    #    block = getattr(pretrained_model, f'b{num_block}')
    #    freeze_module(block)

    trainable_params = sum(p.numel() for p in pretrained_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in pretrained_model.parameters())
    print_and_save(param_log_file_path, f'Number of trainable parameters: {trainable_params}')
    print_and_save(param_log_file_path, f'Total number of parameters: {total_params}\n')

    return pretrained_model

# Function that creates the optimizer with just the trainable parameters (decoder and output layers)
def create_optimizer_for_fine_tuning(param_model, param_learning_rate=HYPERPARAMETERS['init_learning_rate']):
    trainable_parameters = filter(lambda p: p.requires_grad, param_model.parameters())
    return torch.optim.Adam(trainable_parameters, lr=param_learning_rate)

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

    def forward(self, x):
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


def train_step(param_model, param_dataloader, param_optimizer, param_criterion, param_device):
    param_model.train()
    for num_layer in range(4):
        getattr(param_model, f'layer{num_layer}').eval() 
    #for num_block in range(1, 3):
    #    getattr(param_model, f'b{num_block}').eval()
      
    epoch_loss, epoch_jaccard, epoch_dice, epoch_recall, epoch_precision = 0.0, 0.0, 0.0, 0.0, 0.0

    for batched_images, batched_masks in param_dataloader:
        batched_images = batched_images.to(param_device, dtype=torch.float32, non_blocking=True)
        batched_masks = batched_masks.to(param_device, dtype=torch.float32, non_blocking=True)

        param_optimizer.zero_grad()
        y_pred = param_model(batched_images)
        loss = param_criterion(y_pred, batched_masks)
        loss.backward()
        param_optimizer.step()

        epoch_loss += loss.item()

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

def evaluate_step(param_model, param_dataloader, param_criterion, param_device):
    param_model.eval()

    epoch_loss, epoch_jaccard, epoch_dice, epoch_recall, epoch_precision = 0.0, 0.0, 0.0, 0.0, 0.0

    with torch.no_grad():
        for batched_images, batched_masks in param_dataloader:
            batched_images = batched_images.to(param_device, dtype=torch.float32, non_blocking=True)
            batched_masks = batched_masks.to(param_device, dtype=torch.float32, non_blocking=True)

            y_pred = param_model(batched_images)
            loss = param_criterion(y_pred, batched_masks)

            epoch_loss += loss.item()

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
    hyperparameters_log_text = f'Image size: {HYPERPARAMETERS["image_size"]}\nBatch size: {HYPERPARAMETERS["batch_size"]}\nLR: {HYPERPARAMETERS["init_learning_rate"]}\nEpochs: {HYPERPARAMETERS["num_epochs"]}\nScheduler Patience: {HYPERPARAMETERS["scheduler_patience"]}\nEarly Stopping Patience: {HYPERPARAMETERS["early_stopping_patience"]}\n'
    print_and_save(LOG_PATH, hyperparameters_log_text)

    # Load the images and masks file names for training and validation
    (train_images_paths, train_masks_paths), (validation_images_paths, validation_masks_paths) = load_data(DATASET_PATH)
    train_images_paths, train_masks_paths = shuffle_data(train_images_paths, train_masks_paths)
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
    train_dataset = SegmentationDataset(train_images_paths, train_masks_paths, HYPERPARAMETERS['image_size'], param_transform=augmentation)
    validation_dataset = SegmentationDataset(validation_images_paths, validation_masks_paths, HYPERPARAMETERS['image_size'])
    
    # Create dataloaders
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=HYPERPARAMETERS['batch_size'], shuffle=True, num_workers=2, pin_memory=True, persistent_workers=True)
    validation_dataloader = DataLoader(dataset=validation_dataset, batch_size=HYPERPARAMETERS['batch_size'], shuffle=False, num_workers=2, pin_memory=True, persistent_workers=True)

    # Load the pretrained model and freeze the encoder layers and create the optimizer for fine-tuning
    model = load_pretrained_model(PRETRAINED_CHECKPOINT_PATH, DEVICE)
    optimizer = create_optimizer_for_fine_tuning(model, HYPERPARAMETERS['init_learning_rate'])

    # Define the learning rate scheduler and loss criterion
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=HYPERPARAMETERS['scheduler_patience'])
    criterion = DiceBCELoss()

    best_validation_metric = 0.0
    num_epochs_no_improvement = 0

    for epoch in range(HYPERPARAMETERS['num_epochs']):
        start_time = time.time()

        train_loss, train_metrics = train_step(model, train_dataloader, optimizer, criterion, DEVICE)
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
        epoch_log_text = f'Epoch {epoch+1} | Epoch Time: {epoch_duration_min}m {epoch_duration_sec}s\n'
        epoch_log_text += f'\tTrain Loss: {train_loss:.4f} - Jaccard: {train_metrics[0]:.4f} - Dice (F1): {train_metrics[1]:.4f} - Recall: {train_metrics[2]:.4f} - Precision: {train_metrics[3]:.4f}\n'
        epoch_log_text += f'\tValidation Loss: {validation_loss:.4f} - Jaccard: {validation_metrics[0]:.4f} - Dice (F1): {validation_metrics[1]:.4f} - Recall: {validation_metrics[2]:.4f} - Precision: {validation_metrics[3]:.4f}\n'
        print_and_save(LOG_PATH, epoch_log_text)

        if num_epochs_no_improvement == HYPERPARAMETERS['early_stopping_patience']:
            print_and_save(LOG_PATH, f'Early stopping triggered after {epoch+1} epochs.')
            break