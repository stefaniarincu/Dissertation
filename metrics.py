import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


''' ================================= DICE BCE (Segmentation) LOSS FUNCTION ================================= '''

# Class for the standard Dice + BCE loss function used for training the model
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
def compute_metrics(y_true, y_pred):
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

# Helper function that updates the metrics dictionary with the computed metrics for a given pair of true and predicted masks
def update_metrics(results, y_true, y_pred):
    score_jaccard, score_dice, score_recall, score_precision = compute_metrics(y_true, y_pred)
    results['jaccard'] += score_jaccard
    results['dice'] += score_dice
    results['recall'] += score_recall
    results['precision'] += score_precision

def compute_final_results(epoch_loss, results, num_samples):
    epoch_loss /= num_samples
    for key in results:
        results[key] /= num_samples
    return epoch_loss, [results['jaccard'], results['dice'], results['recall'], results['precision']]