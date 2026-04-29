import torch
import torch.nn as nn
import torch.nn.functional as F
from models_tresunet import DecoderBlock


''' ======================================== FUSED TransResUNet MODEL ======================================== '''

# Projection/Alignment block used to align features from the datset specific models before combining them
class ProjectionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.projection_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.projection_block(x)

# Fused TransResUNet model that combines the features from the dataset specific models using cross-attention blocks and residual connections
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

class TResUnetFusedModel(nn.Module):
    def __init__(self, dataset_specific_models):
        super().__init__()

        # Load the pretrained dataset specific models and the weighting mode for combining their features
        self.dataset_specific_1 = dataset_specific_models[0]
        self.dataset_specific_2 = dataset_specific_models[1]
        self.dataset_specific_3 = dataset_specific_models[2]

        # Cross-attention blocks for each encoder level
        '''self.cross_attn1 = CrossAttentionBlock(64)
        self.cross_attn2 = CrossAttentionBlock(256)
        self.cross_attn3 = CrossAttentionBlock(512)'''

        # Projection blocks for s1
        self.proj_s1_1 = ProjectionBlock(64, 64)
        self.proj_s1_2 = ProjectionBlock(64, 64)
        self.proj_s1_3 = ProjectionBlock(64, 64)

        # Projection blocks for s2
        self.proj_s2_1 = ProjectionBlock(256, 256)
        self.proj_s2_2 = ProjectionBlock(256, 256)
        self.proj_s2_3 = ProjectionBlock(256, 256)

        # Projection blocks for s3
        self.proj_s3_1 = ProjectionBlock(512, 512)
        self.proj_s3_2 = ProjectionBlock(512, 512)
        self.proj_s3_3 = ProjectionBlock(512, 512)

        # Projection blocks for bottleneck
        self.proj_b_1 = ProjectionBlock(512, 512)
        self.proj_b_2 = ProjectionBlock(512, 512)
        self.proj_b_3 = ProjectionBlock(512, 512)

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

    def train(self, mode=True):
        super().train(mode)
        # Keep the dataset specific models in evaluation mode to prevent their weights from being updated during training
        self.dataset_specific_1.eval()
        self.dataset_specific_2.eval()
        self.dataset_specific_3.eval()
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
            weights = torch.full((dataset_ids.shape[0], num_dataset_specific_models), 0.25, device=dataset_ids.device, dtype=torch.float32)
            return weights.scatter_(1, dataset_ids.view(-1, 1), 0.5)

    @staticmethod
    def expand_weights(weights):
        w1 = weights[:, 0].view(-1, 1, 1, 1)
        w2 = weights[:, 1].view(-1, 1, 1, 1)
        w3 = weights[:, 2].view(-1, 1, 1, 1)
        return w1, w2, w3
    
    def forward(self, x, dataset_ids=None, weighting_mode=None, return_features=False, return_features_unet=False):
        with torch.no_grad():
            # Encode features from each dataset specific model
            [ds1_s1, ds1_s2, ds1_s3, ds1_b] = self.dataset_specific_1.encode(x)
            [ds2_s1, ds2_s2, ds2_s3, ds2_b] = self.dataset_specific_2.encode(x)
            [ds3_s1, ds3_s2, ds3_s3, ds3_b] = self.dataset_specific_3.encode(x)

        # Cross-attention on encoder outputs
        '''ds1_s1 = self.cross_attn1(ds1_s1, ds2_s1) + self.cross_attn1(ds1_s1, ds3_s1) + self.cross_attn1(ds2_s1, ds3_s1)
        ds1_s2 = self.cross_attn2(ds1_s2, ds2_s2) + self.cross_attn2(ds1_s2, ds3_s2) + self.cross_attn2(ds2_s2, ds3_s2)
        ds1_s3 = self.cross_attn3(ds1_s3, ds2_s3) + self.cross_attn3(ds1_s3, ds3_s3) + self.cross_attn3(ds2_s3, ds3_s3)'''

        # Apply projection blocks
        p1_s1 = self.proj_s1_1(ds1_s1)
        p2_s1 = self.proj_s1_2(ds2_s1)
        p3_s1 = self.proj_s1_3(ds3_s1)

        p1_s2 = self.proj_s2_1(ds1_s2)
        p2_s2 = self.proj_s2_2(ds2_s2)
        p3_s2 = self.proj_s2_3(ds3_s2)

        p1_s3 = self.proj_s3_1(ds1_s3)
        p2_s3 = self.proj_s3_2(ds2_s3)
        p3_s3 = self.proj_s3_3(ds3_s3)

        p1_b = self.proj_b_1(ds1_b)
        p2_b = self.proj_b_2(ds2_b)
        p3_b = self.proj_b_3(ds3_b)

        if weighting_mode is not None and dataset_ids is not None:
            weights = self.compute_dataset_specific_weights(dataset_ids, weighting_mode, 3)
            weights_1, weights_2, weights_3 = self.expand_weights(weights)

            # Concatenate the encoder outputs with cross-attention applied
            combined_s1 = torch.cat([weights_1 * p1_s1, weights_2 * p2_s1, weights_3 * p3_s1], dim=1)
            combined_s2 = torch.cat([weights_1 * p1_s2, weights_2 * p2_s2, weights_3 * p3_s2], dim=1)
            combined_s3 = torch.cat([weights_1 * p1_s3, weights_2 * p2_s3, weights_3 * p3_s3], dim=1)

            # Concatenate bottleneck features from all dataset specific models
            combined_bottleneck = torch.cat((weights_1 * p1_b, weights_2 * p2_b, weights_3 * p3_b), dim=1)
        else:
            # Concatenate the encoder outputs with cross-attention applied
            combined_s1 = torch.cat([p1_s1, p2_s1, p3_s1], dim=1)
            combined_s2 = torch.cat([p1_s2, p2_s2, p3_s2], dim=1)
            combined_s3 = torch.cat([p1_s3, p2_s3, p3_s3], dim=1)

            # Concatenate bottleneck features from all dataset specific models
            combined_bottleneck = torch.cat((p1_b, p2_b, p3_b), dim=1)

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
            return y, [combined_s1, combined_s2, combined_s3]
        return y