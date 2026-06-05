import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from medpy.metric.binary import hd95


''' ================================= DICE BCE (Segmentation) LOSS FUNCTION ================================= '''

# Class for the standard Dice + BCE loss function used for segmentation task when training the models
class DiceBCELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets, smooth=1):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='mean')
        
        inputs = torch.sigmoid(inputs)
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2.0 * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        
        return bce_loss + dice_loss


''' ========================================== COMPUTE METRICS ========================================== '''

# Function that computes the Jaccard, Dice, Recall and Precision metrics for a given pair of true and predicted masks
def compute_metrics(y_true, y_pred, test_mode=False):
    _, height, width = y_true.shape

    y_true = y_true.detach().cpu().numpy()

    # Pass the output through sigmoid and convert to binary mask
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
    
    if test_mode:
        if y_true.sum() == 0 and y_pred.sum() == 0:
            score_hd95 = 0.0
        elif (y_true.sum() == 0 and y_pred.sum() > 0) or (y_true.sum() > 0 and y_pred.sum() == 0):
            score_hd95 = 100.0
        else:
            score_hd95 = hd95(y_pred.reshape(height, width), y_true.reshape(height, width))

        return score_jaccard, score_dice, score_recall, score_precision, score_hd95

    return score_jaccard, score_dice, score_recall, score_precision

# Helper function that updates the accumulated metric sums for a pair of true and predicted masks
def update_metrics(results, y_true, y_pred, test_mode=False):
    if test_mode:
        score_jaccard, score_dice, score_recall, score_precision, score_hd95 = compute_metrics(y_true, y_pred, test_mode=True)
        results['hd95'] += score_hd95
    else:
        score_jaccard, score_dice, score_recall, score_precision = compute_metrics(y_true, y_pred, test_mode=False)

    results['jaccard'] += score_jaccard
    results['dice'] += score_dice
    results['recall'] += score_recall
    results['precision'] += score_precision

# Helper function that computes the final average epoch loss and metrics
def compute_final_results(epoch_loss, results, num_samples, test_mode=False):
    epoch_loss /= num_samples
    for key in results:
        results[key] /= num_samples

    if test_mode:
        return epoch_loss, [results['jaccard'], results['dice'], results['recall'], results['precision'], results['hd95']]

    return epoch_loss, [results['jaccard'], results['dice'], results['recall'], results['precision']]