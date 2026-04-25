import torch
from torch import nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation


''' =================================== LOADING SEGFORMER FROM HUGGING FACE =================================== '''

# Constants for Segformer with mappings for classes and ids to look for
#ID2LABEL = {0: 'background', 1: 'foreground'}
#LABEL2ID = {'background': 0, 'foreground': 1}

# Function that loads a Segformer model with the specified name and configures it for the current segmentation task
def load_segformer(model_name, num_labels=1):
    #hf_segformer_model = SegformerForSemanticSegmentation.from_pretrained(model_name, num_labels=len(ID2LABEL), id2label=ID2LABEL, label2id=LABEL2ID, ignore_mismatched_sizes=True)
    hf_segformer_model = SegformerForSemanticSegmentation.from_pretrained(model_name, num_labels=num_labels, ignore_mismatched_sizes=True)
    return hf_segformer_model


''' ========================================= CREATE SEGFORMER WRAPPER ========================================= '''

# Wrapper class for the Segformer model to enable simpler feature extraction and forward pass for our specific segmentation task
class Segformer(nn.Module):
    def __init__(self, hf_segformer_model):
        super().__init__()

        self.segformer_model = hf_segformer_model
        self.out_channels = list(self.segformer_model.config.hidden_sizes)
    
    @classmethod
    def load_from_pretrained(cls, model_name, num_labels=1):
        #hf_segformer_model = SegformerForSemanticSegmentation.from_pretrained(model_name, num_labels=len(ID2LABEL), id2label=ID2LABEL, label2id=LABEL2ID, ignore_mismatched_sizes=True)
        hf_segformer_model = SegformerForSemanticSegmentation.from_pretrained(model_name, num_labels=num_labels, ignore_mismatched_sizes=True)
        return cls(hf_segformer_model)

    def encode(self, x):
        self.segformer_model.eval()

        with torch.inference_mode():
            # Pass the input through the model to get the output feature maps
            outputs = self.segformer_model(pixel_values=x, output_hidden_states=True)
            hidden_states = list(outputs.hidden_states)
            return hidden_states

    def forward(self, x, return_features=False):
        outputs = self.segformer_model(pixel_values=x, output_hidden_states=return_features)
        logits = F.interpolate(outputs.logits, size=x.shape[2:], mode='bilinear', align_corners=False)

        if return_features:
            hidden_states = list(outputs.hidden_states)
            '''print(outputs.hidden_states[0].shape)
            print(outputs.hidden_states[1].shape)
            print(outputs.hidden_states[2].shape)
            print(outputs.hidden_states[3].shape)'''
            return logits, hidden_states
        
        return logits


''' ===================================== CREATE SEGFORMER FEATURE ADAPTER ===================================== '''

# Feature adapter class for the Segformer model
class SegformerFeatureAdapter(nn.Module):
    def __init__(self, segformer_channels):
        super().__init__()

        s1, s2, s3, s4 = segformer_channels

        self.s1_projection = nn.Sequential(
            nn.Conv2d(s1, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.s2_projection = nn.Sequential(
            nn.Conv2d(s2, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.s3_projection = nn.Sequential(
            nn.Conv2d(s3, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        self.bottleneck_projection = nn.Sequential(
            nn.Conv2d(s4, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x, segformer_features):
        segformer_f1, segformer_f2, segformer_f3, segformer_bottleneck = segformer_features

        h, w = x.shape[2], x.shape[3]

        f1 = F.interpolate(segformer_f1, size=(h // 2, w // 2), mode='bilinear', align_corners=False)
        f2 = F.interpolate(segformer_f2, size=(h // 4, w // 4), mode='bilinear', align_corners=False)
        f3 = F.interpolate(segformer_f3, size=(h // 8, w // 8), mode='bilinear', align_corners=False)
        f4 = F.interpolate(segformer_bottleneck, size=(h // 16, w // 16), mode='bilinear', align_corners=False)

        s1 = self.s1_projection(f1)
        s2 = self.s2_projection(f2)
        s3 = self.s3_projection(f3)
        bottleneck = self.bottleneck_projection(f4)

        return [s1, s2, s3, bottleneck]