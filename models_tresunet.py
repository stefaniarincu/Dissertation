import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from torch.nn import functional as F
from model_segformer import SegformerFeatureAdapter


''' ============================================ ResNet BACKBONE ============================================ '''

# ResNet backbone
model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, 
                     groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super().__init__()

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
        super().__init__()
        
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
                 groups=1, width_per_group=64, replace_stride_with_dilation=None, norm_layer=None):
        super().__init__()
        
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
        
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
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
            layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width, 
                                dilation=self.dilation, norm_layer=norm_layer))

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


''' =========================================== TransResUNet MODEL =========================================== '''

# TransResUNet model
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
        self.r1 = ResidualBlock(in_c[0] + in_c[1], out_c)
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

        # ResNet50 backbone for encoder blocks
        backbone = resnet50()
        self.layer0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.layer1 = nn.Sequential(backbone.maxpool, backbone.layer1)
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3

        # Transformer bottleneck and dilated conv for bottleneck
        self.b1 = BottleneckTResUnet(1024, 256, 256, num_layers=2)
        self.b2 = DilatedConv(1024, 256)

        # Decoder blocks
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
    

''' ======================================== FUSED TransResUNet MODEL ======================================== '''

# Fused TransResUNet model that combines the features from the dataset specific models using cross-attention blocks and residual connections
class ConvolveResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # For matching dimensions in case the input and output channels are different
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1,
                                   padding=0) if in_channels != out_channels else None

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

class TResUnetFusedModel(nn.Module):
    def __init__(self, dataset_specific_models, use_segformers=False):
        super().__init__()
        num_dataset_specific_models = len(dataset_specific_models)

        # Use ModuleList to store the dataset specific models
        self.dataset_specific_models = nn.ModuleList(dataset_specific_models)

        # Set flag if using Segformers to add additional projection layers for feature alignment
        self.use_segformers = use_segformers
        if self.use_segformers:
            self.feature_adapters = nn.ModuleList([
                SegformerFeatureAdapter(model.out_channels) for model in dataset_specific_models
            ])
        
        # Cross-attention blocks for each encoder level
        '''self.cross_attn_s1 = CrossAttentionBlock(64)
        self.cross_attn_s2 = CrossAttentionBlock(256)
        self.cross_attn_s3 = CrossAttentionBlock(512)'''

        # Convolutional blocks for combined encoder outputs 
        # s1 = 64, s2 = 256, s3 = 512, bottleneck = 512 (multiply for concatenation)
        self.conv_1 = ConvolveResidualBlock(512 * num_dataset_specific_models, 512)
        self.conv_2 = ConvolveResidualBlock(512 * num_dataset_specific_models, 512)
        self.conv_3 = ConvolveResidualBlock(256 * num_dataset_specific_models, 256)
        self.conv_4 = ConvolveResidualBlock(64 * num_dataset_specific_models, 64)

        # Decoder blocks
        self.d1 = DecoderBlock([512, 512], 256)
        self.d2 = DecoderBlock([256, 256], 128)
        self.d3 = DecoderBlock([128, 64], 64)
        self.d4 = DecoderBlock([64, 3], 32)

        # Final output layer
        self.output = nn.Conv2d(32, 1, kernel_size=1)

    def train(self, mode=True):
        super().train(mode)
        # Keep the dataset specific models in evaluation mode to prevent their weights from being updated during training
        for model in self.dataset_specific_models:
            model.eval()
        return self
    
    @staticmethod
    # Determine the weights for each dataset specific model based on the specified mode and the dataset ids of the samples in the batch
    def compute_dataset_specific_weights(dataset_ids, weighting_mode, num_dataset_specific_models):
        # one hot encoding = > 1 for the dataset specific model corresponding to the dataset and 0 for the others
        if weighting_mode == 0: 
            return F.one_hot(dataset_ids, num_classes=num_dataset_specific_models).float()
        # uniform weights = > 1/num_dataset_specific_models for all dataset specific models
        elif weighting_mode == 1: 
            return torch.full((dataset_ids.shape[0], num_dataset_specific_models), 1.0 / num_dataset_specific_models, device=dataset_ids.device, dtype=torch.float32)
        # biased weights = > 0.5 for the dataset specific model corresponding to the dataset and 0.25 for the others
        elif weighting_mode == 2:
            if num_dataset_specific_models == 2:
                weights = torch.full((dataset_ids.shape[0], num_dataset_specific_models), 0.25 * num_dataset_specific_models, device=dataset_ids.device, dtype=torch.float32)
                return weights.scatter_(1, dataset_ids.view(-1, 1), 0.75 * num_dataset_specific_models)
            elif num_dataset_specific_models == 3:
                weights = torch.full((dataset_ids.shape[0], num_dataset_specific_models), 0.25 * num_dataset_specific_models, device=dataset_ids.device, dtype=torch.float32)
                return weights.scatter_(1, dataset_ids.view(-1, 1), 0.5 * num_dataset_specific_models)
            elif num_dataset_specific_models == 4:
                weights = torch.full((dataset_ids.shape[0], num_dataset_specific_models), 0.15 * num_dataset_specific_models, device=dataset_ids.device, dtype=torch.float32)
                return weights.scatter_(1, dataset_ids.view(-1, 1), 0.55 * num_dataset_specific_models)

    def forward(self, x, dataset_ids=None, weighting_mode=None, return_features=False, return_features_unet=False):
        num_dataset_specific_models = len(self.dataset_specific_models)

        with torch.no_grad():
            # Encode features from each dataset specific model
            all_features = [model.encode(x) for model in self.dataset_specific_models]

        if self.use_segformers:
            all_features = [self.feature_adapters[i](x, all_features[i]) for i in range(num_dataset_specific_models)]

        # Cross-attention on encoder outputs
        '''cross_attn_s1 = self.cross_attn_s1(all_features[0][0], all_features[1][0])
        cross_attn_s2 = self.cross_attn_s2(all_features[0][1], all_features[1][1])
        cross_attn_s3 = self.cross_attn_s3(all_features[0][2], all_features[1][2])

        for i in range(2, num_dataset_specific_models):
            cross_attn_s1 = cross_attn_s1 + self.cross_attn_s1(all_features[0][0], all_features[i][0])
            cross_attn_s2 = cross_attn_s2 + self.cross_attn_s2(all_features[0][1], all_features[i][1])
            cross_attn_s3 = cross_attn_s3 + self.cross_attn_s3(all_features[0][2], all_features[i][2])

        all_features[0][0] = cross_attn_s1
        all_features[0][1] = cross_attn_s2
        all_features[0][2] = cross_attn_s3'''

        if weighting_mode is not None and dataset_ids is not None:
            weights = self.compute_dataset_specific_weights(dataset_ids, weighting_mode, num_dataset_specific_models)
            weights = [weights[:, i].view(-1, 1, 1, 1) for i in range(weights.shape[1])]

            # Concatenate the encoder outputs with cross-attention applied
            combined_s1 = torch.cat([weights[i] * all_features[i][0] for i in range(num_dataset_specific_models)], dim=1)
            combined_s2 = torch.cat([weights[i] * all_features[i][1] for i in range(num_dataset_specific_models)], dim=1)
            combined_s3 = torch.cat([weights[i] * all_features[i][2] for i in range(num_dataset_specific_models)], dim=1)

            # Concatenate bottleneck features from all dataset specific models
            combined_bottleneck = torch.cat([weights[i] * all_features[i][3] for i in range(num_dataset_specific_models)], dim=1)
        else:
            # Concatenate the encoder outputs with cross-attention applied
            combined_s1 = torch.cat([all_features[i][0] for i in range(num_dataset_specific_models)], dim=1)
            combined_s2 = torch.cat([all_features[i][1] for i in range(num_dataset_specific_models)], dim=1)
            combined_s3 = torch.cat([all_features[i][2] for i in range(num_dataset_specific_models)], dim=1)

            # Concatenate bottleneck features from all dataset specific models
            combined_bottleneck = torch.cat([all_features[i][3] for i in range(num_dataset_specific_models)], dim=1)

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
        if return_features_unet:
            return y, [conv_s1, conv_s2, conv_s3] #[combined_s1, combined_s2, combined_s3]
        return y