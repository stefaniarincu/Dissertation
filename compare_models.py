import os
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import torch
from torch.utils.data import DataLoader
from utils import seed_all
from data import load_split_data, SegmentationDataset
from models_tresunet import TResUnet
from model_unet import UNet
from model_segformer import Segformer


SEED = 42
DEVICE = torch.device('cuda')

# Constant for hyperparameters (moved here for clarity and easy modification)
HYPERPARAMETERS = {
    'image_size': (256, 256),
    'batch_size': 16,
    'segformer_model_name': 'nvidia/mit-b2'
}

# Constant for the root path for all necessary files
ROOT_PATH = '/root/Disertation'

# Constants for dataset name and path
DATASET_NAME = 'kits' # 'bmshare', 'brats'
DATASET_PATH = f'{ROOT_PATH}/datasets/{DATASET_NAME}'

# Constant for dataset specific model checkpoint path 
DATASET_SPECIFIC_MODEL_ROOT_PATH = f'{ROOT_PATH}/files/final_experiments/dataset_specific/fract/unet'
DATASET_SPECIFIC_MODEL_CHECKPOINT = f'{DATASET_SPECIFIC_MODEL_ROOT_PATH}/{DATASET_NAME}/dataset_specific_model_{DATASET_NAME}.pth'

# Constant for the distilled model checkpoint path
DISTILLED_MODEL_CHECKPOINT_ROOT_PATH = f'{ROOT_PATH}/files/final_experiments/knowledge_distillation/kits_lits_lung/fused_from_unets_not_weighted_no_cross_attention_3K_samples_3ds_new_dropout_new_adapter'
DISTILLED_MODEL_CHECKPOINT = f'{DISTILLED_MODEL_CHECKPOINT_ROOT_PATH}/{DATASET_NAME}/distilled_model_{DATASET_NAME}.pth'

# Constant for the path to save the visualizations
VISUALIZATIONS_ROOT_PATH = f'{ROOT_PATH}/files/visualizations/fused_from_unets_not_weighted_no_cross_attention_3K_samples_3ds_new_dropout_new_adapter/{DATASET_NAME}'
os.makedirs(VISUALIZATIONS_ROOT_PATH, exist_ok=True)

# Function that computes the Dice and Jaccard scores for a given pair of true and predicted masks
def compute_dice_jaccard_per_image(y_true, y_pred):
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
    score_dice = (2.0 * intersection + 1e-15) / (y_true.sum() + y_pred.sum() + 1e-15)
    score_jaccard = (intersection + 1e-15) / (union + 1e-15)

    return score_dice, score_jaccard

# Function that evaluates the model on the given dataloader and computes the Dice and Jaccard scores for each image in the dataloader
def evaluate_per_image(model, dataloader, device):
    model.eval()
    dice_scores = []
    jaccard_scores = []
    pred_masks = []
    
    with torch.inference_mode():
        for batched_images, batched_masks in dataloader:
            batched_images = batched_images.to(device, dtype=torch.float32, non_blocking=True)
            batched_masks = batched_masks.to(device, dtype=torch.float32, non_blocking=True)

            y_pred = model(batched_images)
            y_pred_masks = (torch.sigmoid(y_pred) > 0.5).float()

            for yt, yp, yp_mask in zip(batched_masks, y_pred, y_pred_masks):
                score_dice, score_jaccard = compute_dice_jaccard_per_image(yt, yp)

                dice_scores.append(score_dice)
                jaccard_scores.append(score_jaccard)

                y_pred_mask = yp_mask.squeeze().cpu().numpy().astype(np.uint8)
                pred_masks.append(y_pred_mask)

    return dice_scores, jaccard_scores, pred_masks

def plot_per_image_scores(initial_scores, distilled_scores, name, save_path):
    image_indices = np.arange(len(initial_scores))

    plt.figure(figsize=(14, 6))
    plt.plot(image_indices, initial_scores, label='Dataset-Specific model', marker='o', linestyle='', alpha=0.6)
    plt.plot(image_indices, distilled_scores, label='Distilled model', marker='s', linestyle='', alpha=0.6)

    plt.title(f'{name} score distribution')
    plt.xlabel('Image index')
    plt.ylabel(f'{name} per image')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_differences_per_image(initial_scores, distilled_scores, name, save_path):
    differences = np.array(initial_scores) - np.array(distilled_scores)
    image_indices = np.arange(len(differences))

    plt.figure(figsize=(14, 6))
    plt.plot(image_indices, differences, label='Difference (Dataset-Specific - Distilled)', marker='o', linestyle='', alpha=0.6)

    plt.title(f'{name} difference per image')
    plt.xlabel('Image index')
    plt.ylabel(f'{name} difference per image')
    plt.axhline(0, color='gray', linestyle='--', linewidth=1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def write_top_k_differences(initial_scores, distilled_scores, test_images_paths, save_path, k=50):
    differences = np.array(initial_scores) - np.array(distilled_scores)
    top_k_indices = np.argsort(np.abs(differences))[-k:]
    top_k_indices = top_k_indices[np.argsort(np.abs(differences[top_k_indices]))][::-1]

    with open(save_path, 'w') as f:
        f.write('image name\tDice difference (Dataset-Specific - Distilled)\tJaccard difference (Dataset-Specific - Distilled)\n')
        for idx in top_k_indices:
            image_path = test_images_paths[idx]
            image_name = os.path.basename(image_path)
            dice_difference = initial_scores[idx] - distilled_scores[idx]
            jaccard_difference = initial_scores[idx] - distilled_scores[idx]
            f.write(f'{image_name:<30}{abs(dice_difference)*100:<10.2f}{abs(jaccard_difference)*100:<10.2f}\n')

def load_rgb_image(image_path, image_size):
    image = cv.imread(image_path, cv.IMREAD_COLOR)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image = cv.resize(image, image_size, interpolation=cv.INTER_LINEAR)
    return image

def load_binary_mask(mask_path, image_size):
    mask = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)
    mask = cv.resize(mask, image_size, interpolation=cv.INTER_NEAREST)
    mask = (mask > 127).astype(np.uint8)
    return mask

def load_images_and_masks(images_paths, masks_paths, image_size):
    images = []
    gt_masks = []

    for img_path, mask_path in zip(images_paths, masks_paths):
        image = load_rgb_image(img_path, image_size)
        gt_mask = load_binary_mask(mask_path, image_size)

        images.append(image)
        gt_masks.append(gt_mask)

    return images, gt_masks

# Overlay the ground truth mask and the mask predicted by the distilled model
def create_two_masks_overlay(original_image, mask_1, mask_2):
    mask_1 = mask_1.astype(np.uint8)
    mask_2 = mask_2.astype(np.uint8)

    true_positive = np.logical_and(mask_1 == 1, mask_2 == 1)
    false_positive = np.logical_and(mask_1 == 0, mask_2 == 1)
    false_negative = np.logical_and(mask_1 == 1, mask_2 == 0)

    color_mask = np.zeros_like(original_image, dtype=np.uint8)
    color_mask[true_positive] = [0, 255, 0]  # Green
    color_mask[false_positive] = [255, 0, 0]  # Red
    color_mask[false_negative] = [0, 0, 255]  # Blue

    overlay = original_image.copy()
    mask_pixels = true_positive | false_positive | false_negative

    overlay[mask_pixels] = (original_image[mask_pixels] * 0.5 + color_mask[mask_pixels] * 0.5).astype(np.uint8)
    return overlay

def create_binary_mask_visualization(mask):
    binary_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    binary_mask[mask == 1] = [255, 255, 255]
    return binary_mask

def add_separator(height, width, color=(255, 255, 255)):
    separator = np.full((height, width, 3), color, dtype=np.uint8)
    return separator

'''def save_overlay_visualization(images_paths, original_images, mask_1, mask_2, distilled_dice_scores, distilled_jaccard_scores, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    for image_path, original_image, mask_1, mask_2, distilled_dice_score, distilled_jaccard_score in zip(images_paths, original_images, mask_1, mask_2, distilled_dice_scores, distilled_jaccard_scores):
        image_path = os.path.splitext(os.path.basename(image_path))[0]

        overlay = create_two_masks_overlay(original_image, mask_1, mask_2)
        text = f'Dice: {distilled_dice_score*100:.2f}, Jaccard: {distilled_jaccard_score*100:.2f}'

        cv.putText(overlay, text, (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        save_path = os.path.join(save_dir, f'{image_path}.png')
        cv.imwrite(save_path, cv.cvtColor(overlay, cv.COLOR_RGB2BGR))

def save_top_differences(images_paths, original_images, gt_masks, initial_pred_masks, distilled_pred_masks, initial_dice_scores, distilled_dice_scores, initial_jaccard_scores, distilled_jaccard_scores, save_dir, k=50):
    os.makedirs(save_dir, exist_ok=True)

    dice_differences = np.array(initial_dice_scores) - np.array(distilled_dice_scores)
    jaccard_differences = np.array(initial_jaccard_scores) - np.array(distilled_jaccard_scores)

    top_k_indices = np.argsort(np.abs(dice_differences))[-k:]
    top_k_indices = top_k_indices[np.argsort(np.abs(dice_differences[top_k_indices]))][::-1]

    for idx in top_k_indices:
        image_path = os.path.splitext(os.path.basename(images_paths[idx]))[0]
        original_image = original_images[idx]
        gt_mask = gt_masks[idx]
        initial_pred_mask = initial_pred_masks[idx]
        distilled_pred_mask = distilled_pred_masks[idx]
        dice_difference = dice_differences[idx]
        jaccard_difference = jaccard_differences[idx]

        gt = create_binary_mask_visualization(gt_mask)
        initial_pred = create_binary_mask_visualization(initial_pred_mask)
        distilled_pred = create_binary_mask_visualization(distilled_pred_mask)

        cv.putText(gt, f'GT', (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv.putText(initial_pred, f'Dataset-Specific', (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv.putText(initial_pred, f'Dice: {initial_dice_scores[idx]*100:.2f}', (10, 40), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv.putText(initial_pred, f'Jaccard: {initial_jaccard_scores[idx]*100:.2f}', (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv.putText(distilled_pred, f'Distilled', (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv.putText(distilled_pred, f'Dice: {distilled_dice_scores[idx]*100:.2f}', (10, 40), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv.putText(distilled_pred, f'Jaccard: {distilled_jaccard_scores[idx]*100:.2f}', (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        separator = add_separator(original_image.shape[0], 10)
        combined_image = np.hstack((original_image, gt, initial_pred, distilled_pred, separator))
        save_path = os.path.join(save_dir, f'{image_path}.png')
        cv.imwrite(save_path, cv.cvtColor(combined_image, cv.COLOR_RGB2BGR))'''

def save_overlay_visualization(images_paths, original_images, mask_1, mask_2, distilled_dice_scores, distilled_jaccard_scores, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    legend = [
        Patch(color=(0, 1, 0), label='True Positive'),
        Patch(color=(1, 0, 0), label='False Positive'),
        Patch(color=(0, 0, 1), label='False Negative')
    ]

    for image_path, original_image, mask_1, mask_2, distilled_dice_score, distilled_jaccard_score in zip(images_paths, original_images, mask_1, mask_2, distilled_dice_scores, distilled_jaccard_scores):
        if distilled_dice_score*100 != 100.0 or distilled_jaccard_score*100 != 100.0:
            image_path = os.path.splitext(os.path.basename(image_path))[0]

            overlay = create_two_masks_overlay(original_image, mask_1, mask_2)
            text = f'Student vs Ground truth\nDice: {distilled_dice_score*100:.2f}, Jaccard: {distilled_jaccard_score*100:.2f}'
            
            fig, ax = plt.subplots(figsize=(8, 7))
            ax.imshow(overlay)
            ax.set_title(text, pad=10)
            ax.axis('off')

            fig.subplots_adjust(left=0.08, right=0.92, top=0.88, bottom=0.12)
            fig.legend(handles=legend, loc='lower center', bbox_to_anchor=(0.5, 0.03), ncol=3, frameon=True, fontsize=10)
            save_path = os.path.join(save_dir, f'{image_path}.png')
            fig.savefig(save_path)
            plt.close(fig)

def save_overlay_visualization_top(images_paths, original_images, mask_1, mask_2, distilled_dice_scores, distilled_jaccard_scores, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    legend = [
        Patch(color=(0, 1, 0), label='True Positive'),
        Patch(color=(1, 0, 0), label='False Positive'),
        Patch(color=(0, 0, 1), label='False Negative')
    ]

    for image_path, original_image, mask_1, mask_2, distilled_dice_score, distilled_jaccard_score in zip(images_paths, original_images, mask_1, mask_2, distilled_dice_scores, distilled_jaccard_scores):
        #if distilled_dice_score*100 != 100.0 or distilled_jaccard_score*100 != 100.0:
        image_path = os.path.splitext(os.path.basename(image_path))[0]

        overlay = create_two_masks_overlay(original_image, mask_1, mask_2)
        text = f'Student vs Ground truth\nDice: {distilled_dice_score*100:.2f}, Jaccard: {distilled_jaccard_score*100:.2f}'
        
        fig, ax = plt.subplots(figsize=(8, 7))
        ax.imshow(overlay)
        ax.set_title(text, pad=10)
        ax.axis('off')

        fig.subplots_adjust(left=0.08, right=0.92, top=0.88, bottom=0.12)
        fig.legend(handles=legend, loc='lower center', bbox_to_anchor=(0.5, 0.03), ncol=3, frameon=True, fontsize=10)
        save_path = os.path.join(save_dir, f'{image_path}.png')
        fig.savefig(save_path)
        plt.close(fig)

def save_top_differences(images_paths, original_images, gt_masks, initial_pred_masks, distilled_pred_masks, initial_dice_scores, distilled_dice_scores, initial_jaccard_scores, distilled_jaccard_scores, save_dir, k=50):
    os.makedirs(save_dir, exist_ok=True)
    
    overlays_save_dir = os.path.join(save_dir, 'overlays')
    os.makedirs(overlays_save_dir, exist_ok=True)

    dice_differences = np.array(initial_dice_scores) - np.array(distilled_dice_scores)
    jaccard_differences = np.array(initial_jaccard_scores) - np.array(distilled_jaccard_scores)

    top_k_indices = np.argsort(np.abs(dice_differences))[-k:]
    top_k_indices = top_k_indices[np.argsort(np.abs(dice_differences[top_k_indices]))][::-1]

    for idx in top_k_indices:
        image_path = os.path.splitext(os.path.basename(images_paths[idx]))[0]
        original_image = original_images[idx]
        gt_mask = gt_masks[idx]
        initial_pred_mask = initial_pred_masks[idx]
        distilled_pred_mask = distilled_pred_masks[idx]
        dice_difference = dice_differences[idx]
        jaccard_difference = jaccard_differences[idx]

        gt = create_binary_mask_visualization(gt_mask)
        initial_pred = create_binary_mask_visualization(initial_pred_mask)
        distilled_pred = create_binary_mask_visualization(distilled_pred_mask)

        save_overlay_visualization_top([images_paths[idx]], [original_image], [gt_mask], [distilled_pred_mask], [distilled_dice_scores[idx]], [distilled_jaccard_scores[idx]], overlays_save_dir)

        fig, axes = plt.subplots(1, 4, figsize=(18, 6))
        axes[0].imshow(original_image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        axes[1].imshow(gt)
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')

        axes[2].imshow(initial_pred)
        axes[2].set_title(f'Baseline')
        axes[2].axis('off')

        axes[3].imshow(distilled_pred)
        axes[3].set_title(f'Student')
        axes[3].axis('off')

        save_path = os.path.join(save_dir, f'{image_path}.png')
        plt.savefig(save_path)
        plt.close(fig)

def plot_for_one_image(
    original_image,
    gt_mask,
    pred_mask_dataset_specific,
    pred_mask_distilled,
    save_path
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, axes = plt.subplots(
        1, 6,
        figsize=(12, 2.5),
        gridspec_kw={
            'width_ratios': [1, 1, 1, 1, 1, 0.25],
            'wspace': 0.02
        }
    )

    # imagine originală
    axes[0].imshow(original_image)
    axes[0].set_title("Original image", fontsize=8)
    axes[0].axis("off")

    # GT
    axes[1].imshow(gt_mask, cmap='gray')
    axes[1].set_title("Ground truth", fontsize=8)
    axes[1].axis("off")

    # baseline
    axes[2].imshow(pred_mask_dataset_specific, cmap='gray')
    axes[2].set_title("Baseline", fontsize=8)
    axes[2].axis("off")

    # student
    axes[3].imshow(pred_mask_distilled, cmap='gray')
    axes[3].set_title("Student", fontsize=8)
    axes[3].axis("off")

    # overlay student vs GT
    overlay = create_two_masks_overlay(
        original_image,
        gt_mask,
        pred_mask_distilled
    )

    axes[4].imshow(overlay)
    axes[4].set_title("Student vs Ground truth", fontsize=8)
    axes[4].axis("off")

    legend = [
        Patch(color=(0,1,0), label='TP'),
        Patch(color=(1,0,0), label='FP'),
        Patch(color=(0,0,1), label='FN')
    ]

    axes[5].axis('off')

    axes[5].legend(
        handles=legend,
        loc='center',
        frameon=False,
        fontsize=8
    )

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches='tight',
        pad_inches=0.1
    )
    plt.close(fig)

'''def plot_all_for_one_image(plots, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, axes = plt.subplots(
        3, 5,
        figsize=(12, 10),
        constrained_layout=False,
        gridspec_kw={
            "wspace": 0.03,
            "hspace": 0.03
        }
    )

    titles = [
        "Original image",
        "Ground truth",
        "Baseline",
        "Student",
        "Student vs Ground truth"
    ]

    legend = [
        Patch(color=(0, 1, 0), label="True positive"),
        Patch(color=(1, 0, 0), label="False positive"),
        Patch(color=(0, 0, 1), label="False negative")
    ]

    for row, (
        original_image,
        gt_mask,
        pred_mask_dataset_specific,
        pred_mask_distilled
    ) in enumerate(plots):

        error_map = create_two_masks_overlay(
            original_image,
            gt_mask,
            pred_mask_distilled
        )

        images = [
            original_image,
            gt_mask,
            pred_mask_dataset_specific,
            pred_mask_distilled,
            error_map
        ]

        cmaps = [None, "gray", "gray", "gray", None]

        for col in range(5):
            axes[row, col].imshow(images[col], cmap=cmaps[col], aspect='equal')
            axes[row, col].axis("off")

            if row == 0:
                axes[row, col].set_title(
                    titles[col],
                    fontsize=12,
                    pad=6
                )

    fig.legend(
        handles=legend,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.085),
        ncol=3,
        frameon=True,
        fontsize=9
    )

    plt.subplots_adjust(
        left=0.05,      # era 0.02
        right=0.95,     # era 0.98
        top=0.90,       # era 0.92
        bottom=0.10,    # era 0.08
        wspace=0.03,
        hspace=0.005    # era 0.03
    )
    
    fig.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
        #pad_inches=0.05
    )

    plt.close(fig)'''

'''def plot_all_for_one_image(plots, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, axes = plt.subplots(
        4, 5,
        figsize=(13, 12),
        constrained_layout=False,
        gridspec_kw={"wspace": 0.03, "hspace": 0.03}
    )

    titles = [
        "Original image",
        "Ground truth",
        "Baseline",
        "Student",
        "Student vs Ground truth"
    ]

    row_labels = ["BrainMetShare", "BraTS", "BraTS-PED", "ISLES"]

    legend = [
        Patch(color=(0, 1, 0), label="True positive"),
        Patch(color=(1, 0, 0), label="False positive"),
        Patch(color=(0, 0, 1), label="False negative")
    ]

    for row, (
        original_image,
        gt_mask,
        pred_mask_dataset_specific,
        pred_mask_distilled
    ) in enumerate(plots):

        error_map = create_two_masks_overlay(
            original_image,
            gt_mask,
            pred_mask_distilled
        )

        images = [
            original_image,
            gt_mask,
            pred_mask_dataset_specific,
            pred_mask_distilled,
            error_map
        ]

        cmaps = [None, "gray", "gray", "gray", None]

        for col in range(5):
            axes[row, col].imshow(images[col], cmap=cmaps[col], aspect="equal")
            axes[row, col].axis("off")

            if row == 0:
                axes[row, col].set_title(titles[col], fontsize=12, pad=6)

    plt.subplots_adjust(
        left=0.10,
        right=0.95,
        top=0.90,
        bottom=0.10,
        wspace=0.03,
        hspace=0.005
    )

    for row, label in enumerate(row_labels):
        pos = axes[row, 0].get_position()

        fig.text(
            0.09,
            pos.y0 + pos.height / 2,
            label,
            fontsize=12,
            ha="right",
            va="center"
        )

    fig.legend(
        handles=legend,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.055),
        ncol=3,
        frameon=True,
        fontsize=9
    )

    fig.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)'''
    

def plot_all_for_one_image(plots, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, axes = plt.subplots(
        3, 5,
        figsize=(13, 9),
        constrained_layout=False,
        gridspec_kw={"wspace": 0.03, "hspace": 0.03}
    )

    titles = [
        "Original image",
        "Ground truth",
        "Baseline",
        "Student",
        "Student vs Ground truth"
    ]

    row_labels = ["KiTS", "LiTS", "Lung MSD"]

    legend = [
        Patch(color=(0, 1, 0), label="True positive"),
        Patch(color=(1, 0, 0), label="False positive"),
        Patch(color=(0, 0, 1), label="False negative")
    ]

    for row, (
        original_image,
        gt_mask,
        pred_mask_dataset_specific,
        pred_mask_distilled
    ) in enumerate(plots):

        error_map = create_two_masks_overlay(
            original_image,
            gt_mask,
            pred_mask_distilled
        )

        images = [
            original_image,
            gt_mask,
            pred_mask_dataset_specific,
            pred_mask_distilled,
            error_map
        ]

        cmaps = [None, "gray", "gray", "gray", None]

        for col in range(5):
            axes[row, col].imshow(images[col], cmap=cmaps[col], aspect="equal")
            axes[row, col].axis("off")

            if row == 0:
                axes[row, col].set_title(titles[col], fontsize=12, pad=6)

    plt.subplots_adjust(
        left=0.10,
        right=0.95,
        top=0.90,
        bottom=0.10,
        wspace=0.03,
        hspace=0.005
    )

    for row, label in enumerate(row_labels):
        pos = axes[row, 0].get_position()

        fig.text(
            0.09,
            pos.y0 + pos.height / 2,
            label,
            fontsize=12,
            ha="right",
            va="center"
        )

    fig.legend(
        handles=legend,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.055),
        ncol=3,
        frameon=True,
        fontsize=9
    )

    fig.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

if __name__ == '__main__':
    seed_all(SEED)

    # Load test images and masks paths and create dataloader for the test set
    test_images_paths, test_masks_paths = load_split_data(DATASET_PATH, 'test.txt')
    test_dataset = SegmentationDataset(test_images_paths, test_masks_paths, HYPERPARAMETERS['image_size'], transform=None)
    test_dataloader = DataLoader(test_dataset, batch_size=HYPERPARAMETERS['batch_size'], shuffle=False, num_workers=0, pin_memory=True)

    
    distilled_checkpoint_1 = f'{DISTILLED_MODEL_CHECKPOINT_ROOT_PATH}/kits/distilled_model_kits.pth'
    distilled_model_1 = UNet(3, 1, True).to(DEVICE)
    distilled_model_1.load_state_dict(torch.load(distilled_checkpoint_1, map_location=DEVICE))

    specific_checkpoint_1 = f'{DATASET_SPECIFIC_MODEL_ROOT_PATH}/kits/dataset_specific_model_kits.pth'
    dataset_specific_model_1 = UNet(3, 1, True).to(DEVICE)
    dataset_specific_model_1.load_state_dict(torch.load(specific_checkpoint_1, map_location=DEVICE))

    distilled_checkpoint_2 = f'{DISTILLED_MODEL_CHECKPOINT_ROOT_PATH}/lits/distilled_model_lits.pth'
    distilled_model_2 = UNet(3, 1, True).to(DEVICE)
    distilled_model_2.load_state_dict(torch.load(distilled_checkpoint_2, map_location=DEVICE))

    specific_checkpoint_2 = f'{DATASET_SPECIFIC_MODEL_ROOT_PATH}/lits/dataset_specific_model_lits.pth'
    dataset_specific_model_2 = UNet(3, 1, True).to(DEVICE)
    dataset_specific_model_2.load_state_dict(torch.load(specific_checkpoint_2, map_location=DEVICE))

    distilled_checkpoint_3 = f'{DISTILLED_MODEL_CHECKPOINT_ROOT_PATH}/lung/distilled_model_lung.pth'
    distilled_model_3 = UNet(3, 1, True).to(DEVICE)
    distilled_model_3.load_state_dict(torch.load(distilled_checkpoint_3, map_location=DEVICE))

    specific_checkpoint_3 = f'{DATASET_SPECIFIC_MODEL_ROOT_PATH}/lung/dataset_specific_model_lung.pth'
    dataset_specific_model_3 = UNet(3, 1, True).to(DEVICE)
    dataset_specific_model_3.load_state_dict(torch.load(specific_checkpoint_3, map_location=DEVICE))

    image_kits = 'volume-00042_277.png'
    image_lits = 'volume-35_54.png'
    image_lung = 'volume-074_314.png'
  
    '''
    distilled_checkpoint_1 = f'{DISTILLED_MODEL_CHECKPOINT_ROOT_PATH}/kits/distilled_model_kits.pth'
    distilled_model_1 = Segformer.load_from_pretrained(HYPERPARAMETERS['segformer_model_name'], num_labels=1).to(DEVICE)
    distilled_model_1.load_state_dict(torch.load(distilled_checkpoint_1, map_location=DEVICE))

    specific_checkpoint_1 = f'{DATASET_SPECIFIC_MODEL_ROOT_PATH}/kits/dataset_specific_model_kits.pth'
    dataset_specific_model_1 = Segformer.load_from_pretrained(HYPERPARAMETERS['segformer_model_name'], num_labels=1).to(DEVICE)
    dataset_specific_model_1.load_state_dict(torch.load(specific_checkpoint_1, map_location=DEVICE))

    distilled_checkpoint_2 = f'{DISTILLED_MODEL_CHECKPOINT_ROOT_PATH}/lits/distilled_model_lits.pth'
    distilled_model_2 = Segformer.load_from_pretrained(HYPERPARAMETERS['segformer_model_name'], num_labels=1).to(DEVICE)
    distilled_model_2.load_state_dict(torch.load(distilled_checkpoint_2, map_location=DEVICE))

    specific_checkpoint_2 = f'{DATASET_SPECIFIC_MODEL_ROOT_PATH}/lits/dataset_specific_model_lits.pth'
    dataset_specific_model_2 = Segformer.load_from_pretrained(HYPERPARAMETERS['segformer_model_name'], num_labels=1).to(DEVICE)
    dataset_specific_model_2.load_state_dict(torch.load(specific_checkpoint_2, map_location=DEVICE))

    distilled_checkpoint_3 = f'{DISTILLED_MODEL_CHECKPOINT_ROOT_PATH}/lung/distilled_model_lung.pth'
    distilled_model_3 = Segformer.load_from_pretrained(HYPERPARAMETERS['segformer_model_name'], num_labels=1).to(DEVICE)
    distilled_model_3.load_state_dict(torch.load(distilled_checkpoint_3, map_location=DEVICE))

    specific_checkpoint_3 = f'{DATASET_SPECIFIC_MODEL_ROOT_PATH}/lung/dataset_specific_model_lung.pth'
    dataset_specific_model_3 = Segformer.load_from_pretrained(HYPERPARAMETERS['segformer_model_name'], num_labels=1).to(DEVICE)
    dataset_specific_model_3.load_state_dict(torch.load(specific_checkpoint_3, map_location=DEVICE))

    image_kits = 'volume-00018_64.png'
    image_lits = 'volume-16_360.png'
    image_lung = 'volume-033_147.png'
    '''


    plots = []

    for dataset, image_name, specific_model, distilled_model in zip(
        ['kits', 'lits', 'lung'],
        [image_kits, image_lits, image_lung],
        [dataset_specific_model_1, dataset_specific_model_2, dataset_specific_model_3],
        [distilled_model_1, distilled_model_2, distilled_model_3]
    ):
        image_path = os.path.join(f'{ROOT_PATH}/datasets/{dataset}', 'images', image_name)
        mask_path = os.path.join(f'{ROOT_PATH}/datasets/{dataset}', 'masks', image_name)

        original_image = load_rgb_image(image_path, HYPERPARAMETERS['image_size'])
        gt_mask = load_binary_mask(mask_path, HYPERPARAMETERS['image_size'])

        pred_mask_dataset_specific = evaluate_per_image(specific_model, DataLoader(SegmentationDataset([image_path], [mask_path], HYPERPARAMETERS['image_size'], transform=None), batch_size=1), DEVICE)[2][0]
        pred_mask_distilled = evaluate_per_image(distilled_model, DataLoader(SegmentationDataset([image_path], [mask_path], HYPERPARAMETERS['image_size'], transform=None), batch_size=1), DEVICE)[2][0]

        plots.append((original_image, gt_mask, pred_mask_dataset_specific, pred_mask_distilled))

    plot_all_for_one_image(plots, f'{VISUALIZATIONS_ROOT_PATH}/comparison_all.png')

    '''distilled_checkpoint_1 = f'{DISTILLED_MODEL_CHECKPOINT_ROOT_PATH}/bmshare/distilled_model_bmshare.pth'
    distilled_model_1 = TResUnet().to(DEVICE)
    distilled_model_1.load_state_dict(torch.load(distilled_checkpoint_1, map_location=DEVICE))

    specific_checkpoint_1 = f'{DATASET_SPECIFIC_MODEL_ROOT_PATH}/bmshare/dataset_specific_model_bmshare.pth'
    dataset_specific_model_1 = TResUnet().to(DEVICE)
    dataset_specific_model_1.load_state_dict(torch.load(specific_checkpoint_1, map_location=DEVICE))

    distilled_checkpoint_2 = f'{DISTILLED_MODEL_CHECKPOINT_ROOT_PATH}/brats/distilled_model_brats.pth'
    distilled_model_2 = TResUnet().to(DEVICE)
    distilled_model_2.load_state_dict(torch.load(distilled_checkpoint_2, map_location=DEVICE))

    specific_checkpoint_2 = f'{DATASET_SPECIFIC_MODEL_ROOT_PATH}/brats/dataset_specific_model_brats.pth'
    dataset_specific_model_2 = TResUnet().to(DEVICE)
    dataset_specific_model_2.load_state_dict(torch.load(specific_checkpoint_2, map_location=DEVICE))

    distilled_checkpoint_3 = f'{DISTILLED_MODEL_CHECKPOINT_ROOT_PATH}/isles/distilled_model_isles.pth'
    distilled_model_3 = TResUnet().to(DEVICE)
    distilled_model_3.load_state_dict(torch.load(distilled_checkpoint_3, map_location=DEVICE))

    specific_checkpoint_3 = f'{DATASET_SPECIFIC_MODEL_ROOT_PATH}/isles/dataset_specific_model_isles.pth'
    dataset_specific_model_3 = TResUnet().to(DEVICE)
    dataset_specific_model_3.load_state_dict(torch.load(specific_checkpoint_3, map_location=DEVICE))

    image_bmshare = 'XJbcYPzdIeEIqJl.png'
    image_brats = 'BraTS20_Training_352_77.png'
    image_isles = 'sub-strokecase0222_9.png'


    plots = []

    for dataset, image_name, specific_model, distilled_model in zip(
        ['bmshare', 'brats', 'isles'],
        [image_bmshare, image_brats, image_isles],
        [dataset_specific_model_1, dataset_specific_model_2, dataset_specific_model_3],
        [distilled_model_1, distilled_model_2, distilled_model_3]
    ):
        image_path = os.path.join(f'{ROOT_PATH}/datasets/{dataset}', 'images', image_name)
        mask_path = os.path.join(f'{ROOT_PATH}/datasets/{dataset}', 'masks', image_name)

        original_image = load_rgb_image(image_path, HYPERPARAMETERS['image_size'])
        gt_mask = load_binary_mask(mask_path, HYPERPARAMETERS['image_size'])

        pred_mask_dataset_specific = evaluate_per_image(specific_model, DataLoader(SegmentationDataset([image_path], [mask_path], HYPERPARAMETERS['image_size'], transform=None), batch_size=1), DEVICE)[2][0]
        pred_mask_distilled = evaluate_per_image(distilled_model, DataLoader(SegmentationDataset([image_path], [mask_path], HYPERPARAMETERS['image_size'], transform=None), batch_size=1), DEVICE)[2][0]

        plots.append((original_image, gt_mask, pred_mask_dataset_specific, pred_mask_distilled))

    plot_all_for_one_image(plots, f'{VISUALIZATIONS_ROOT_PATH}/comparison_all.png')'''
    
    '''distilled_checkpoint_1 = f'{DISTILLED_MODEL_CHECKPOINT_ROOT_PATH}/bmshare/distilled_model_bmshare.pth'
    distilled_model_1 = TResUnet().to(DEVICE)
    distilled_model_1.load_state_dict(torch.load(distilled_checkpoint_1, map_location=DEVICE))

    specific_checkpoint_1 = f'{DATASET_SPECIFIC_MODEL_ROOT_PATH}/bmshare/dataset_specific_model_bmshare.pth'
    dataset_specific_model_1 = TResUnet().to(DEVICE)
    dataset_specific_model_1.load_state_dict(torch.load(specific_checkpoint_1, map_location=DEVICE))

    distilled_checkpoint_2 = f'{DISTILLED_MODEL_CHECKPOINT_ROOT_PATH}/brats/distilled_model_brats.pth'
    distilled_model_2 = TResUnet().to(DEVICE)
    distilled_model_2.load_state_dict(torch.load(distilled_checkpoint_2, map_location=DEVICE))

    specific_checkpoint_2 = f'{DATASET_SPECIFIC_MODEL_ROOT_PATH}/brats/dataset_specific_model_brats.pth'
    dataset_specific_model_2 = TResUnet().to(DEVICE)
    dataset_specific_model_2.load_state_dict(torch.load(specific_checkpoint_2, map_location=DEVICE))

    distilled_checkpoint_3 = f'{DISTILLED_MODEL_CHECKPOINT_ROOT_PATH}/brats_ped/distilled_model_brats_ped.pth'
    distilled_model_3 = TResUnet().to(DEVICE)
    distilled_model_3.load_state_dict(torch.load(distilled_checkpoint_3, map_location=DEVICE))

    specific_checkpoint_3 = f'{DATASET_SPECIFIC_MODEL_ROOT_PATH}/brats_ped/dataset_specific_model_brats_ped.pth'
    dataset_specific_model_3 = TResUnet().to(DEVICE)
    dataset_specific_model_3.load_state_dict(torch.load(specific_checkpoint_3, map_location=DEVICE))

    distilled_checkpoint_4 = f'{DISTILLED_MODEL_CHECKPOINT_ROOT_PATH}/isles/distilled_model_isles.pth'
    distilled_model_4 = TResUnet().to(DEVICE)
    distilled_model_4.load_state_dict(torch.load(distilled_checkpoint_4, map_location=DEVICE))

    specific_checkpoint_4 = f'{DATASET_SPECIFIC_MODEL_ROOT_PATH}/isles/dataset_specific_model_isles.pth'
    dataset_specific_model_4 = TResUnet().to(DEVICE)
    dataset_specific_model_4.load_state_dict(torch.load(specific_checkpoint_4, map_location=DEVICE))

    image_bmshare = 'yWgLRErStEMrZtm.png'
    image_brats = 'BraTS20_Training_337_63.png'
    image_brats_ped = 'BraTS-PED-00046-000_56.png'
    image_isles = 'sub-strokecase0230_16.png'


    plots = []

    for dataset, image_name, specific_model, distilled_model in zip(
        ['bmshare', 'brats', 'brats_ped', 'isles'],
        [image_bmshare, image_brats, image_brats_ped, image_isles],
        [dataset_specific_model_1, dataset_specific_model_2, dataset_specific_model_3, dataset_specific_model_4],
        [distilled_model_1, distilled_model_2, distilled_model_3, distilled_model_4]
    ):
        image_path = os.path.join(f'{ROOT_PATH}/datasets/{dataset}', 'images', image_name)
        mask_path = os.path.join(f'{ROOT_PATH}/datasets/{dataset}', 'masks', image_name)

        original_image = load_rgb_image(image_path, HYPERPARAMETERS['image_size'])
        gt_mask = load_binary_mask(mask_path, HYPERPARAMETERS['image_size'])

        pred_mask_dataset_specific = evaluate_per_image(specific_model, DataLoader(SegmentationDataset([image_path], [mask_path], HYPERPARAMETERS['image_size'], transform=None), batch_size=1), DEVICE)[2][0]
        pred_mask_distilled = evaluate_per_image(distilled_model, DataLoader(SegmentationDataset([image_path], [mask_path], HYPERPARAMETERS['image_size'], transform=None), batch_size=1), DEVICE)[2][0]

        plots.append((original_image, gt_mask, pred_mask_dataset_specific, pred_mask_distilled))

    plot_all_for_one_image(plots, f'{VISUALIZATIONS_ROOT_PATH}/comparison_all.png')'''

    # Load dataset-specific model
    """dataset_specific_model = TResUnet().to(DEVICE)
    #dataset_specific_model = UNet(3, 1, True).to(DEVICE)
    #dataset_specific_model = Segformer.load_from_pretrained(HYPERPARAMETERS['segformer_model_name'], num_labels=1).to(DEVICE)
    dataset_specific_model.load_state_dict(torch.load(DATASET_SPECIFIC_MODEL_CHECKPOINT, map_location=DEVICE))

    # Load distilled model
    distilled_model = TResUnet().to(DEVICE)
    #distilled_model = UNet(3, 1, True).to(DEVICE)
    #distilled_model = Segformer.load_from_pretrained(HYPERPARAMETERS['segformer_model_name'], num_labels=1).to(DEVICE)
    distilled_model.load_state_dict(torch.load(DISTILLED_MODEL_CHECKPOINT, map_location=DEVICE))

    #original_images_names = ['volume-00042_277.png', 'volume-00131_48.png', 'volume-00131_49.png']
    #original_images_names = ['volume-35_54.png', 'volume-46_39.png', 'volume-84_432.png', 'volume-94_534.png']
    #original_images_names = ['volume-033_141.png', 'volume-042_77.png', 'volume-055_154.png', 'volume-074_314.png', 'volume-074_319.png']
    ''' for original_image_name in original_images_names:
        image_path = os.path.join(DATASET_PATH, 'images', original_image_name)
        mask_path = os.path.join(DATASET_PATH, 'masks', original_image_name)

        original_image = load_rgb_image(image_path, HYPERPARAMETERS['image_size'])
        gt_mask = load_binary_mask(mask_path, HYPERPARAMETERS['image_size'])

        pred_mask_dataset_specific = evaluate_per_image(dataset_specific_model, DataLoader(SegmentationDataset([image_path], [mask_path], HYPERPARAMETERS['image_size'], transform=None), batch_size=1), DEVICE)[2][0]
        pred_mask_distilled = evaluate_per_image(distilled_model, DataLoader(SegmentationDataset([image_path], [mask_path], HYPERPARAMETERS['image_size'], transform=None), batch_size=1), DEVICE)[2][0]

        plot_for_one_image(original_image, gt_mask, pred_mask_dataset_specific, pred_mask_distilled, f'{VISUALIZATIONS_ROOT_PATH}/{os.path.splitext(original_image_name)[0]}_comparison.png')'''

    # Evaluate dataset-specific model
    initial_dice_scores, initial_jaccard_scores, initial_pred_masks = evaluate_per_image(dataset_specific_model, test_dataloader, DEVICE)

    # Evaluate distilled model
    distilled_dice_scores, distilled_jaccard_scores, distilled_pred_masks = evaluate_per_image(distilled_model, test_dataloader, DEVICE)

    # Write to file the Dice and Jaccard scores for both models for each image in the test set
    '''with open(os.path.join(VISUALIZATIONS_ROOT_PATH, f'{DATASET_NAME}_scores_dice.txt'), 'w') as f:
        f.write('image name\tDataset-Specific dice\tDistilled dice\n')
        for image_path, initial_score, distilled_score in zip(test_images_paths, initial_dice_scores, distilled_dice_scores):
            image_name = os.path.basename(image_path)
            f.write(f'{image_name:<30}{initial_score*100:<10.2f}{distilled_score*100:<10.2f}\n')

    with open(os.path.join(VISUALIZATIONS_ROOT_PATH, f'{DATASET_NAME}_scores_jaccard.txt'), 'w') as f:
        f.write('image name\tDataset-Specific jaccard\tDistilled jaccard\n')
        for image_path, initial_score, distilled_score in zip(test_images_paths, initial_jaccard_scores, distilled_jaccard_scores):
            image_name = os.path.basename(image_path)
            f.write(f'{image_name:<30}{initial_score*100:<10.2f}{distilled_score*100:<10.2f}\n')

    with open(os.path.join(VISUALIZATIONS_ROOT_PATH, f'{DATASET_NAME}_scores_differences.txt'), 'w') as f:
        f.write('image name\tDice difference (Dataset-Specific - Distilled)\tJaccard difference (Dataset-Specific - Distilled)\n')
        for image_path, initial_dice, distilled_dice, initial_jaccard, distilled_jaccard in zip(test_images_paths, initial_dice_scores, distilled_dice_scores, initial_jaccard_scores, distilled_jaccard_scores):
            image_name = os.path.basename(image_path)
            dice_difference = initial_dice - distilled_dice
            jaccard_difference = initial_jaccard - distilled_jaccard
            f.write(f'{image_name:<30}{abs(dice_difference)*100:<10.2f}{abs(jaccard_difference)*100:<10.2f}\n')'''

    '''# Plot distribution of Dice and Jaccard scores for both models
    plot_per_image_scores(initial_dice_scores, distilled_dice_scores, 'Dice', f'{VISUALIZATIONS_ROOT_PATH}/{DATASET_NAME}_dice_distribution.png')
    plot_per_image_scores(initial_jaccard_scores, distilled_jaccard_scores, 'Jaccard', f'{VISUALIZATIONS_ROOT_PATH}/{DATASET_NAME}_jaccard_distribution.png')

    # Plot differences in Dice and Jaccard scores between the two models for each image
    plot_differences_per_image(initial_dice_scores, distilled_dice_scores, 'Dice', f'{VISUALIZATIONS_ROOT_PATH}/{DATASET_NAME}_dice_difference.png')
    plot_differences_per_image(initial_jaccard_scores, distilled_jaccard_scores, 'Jaccard', f'{VISUALIZATIONS_ROOT_PATH}/{DATASET_NAME}_jaccard_difference.png')'''

    # Write to file the top 50 images with the largest absolute differences in Dice and Jaccard scores between the two models
    write_top_k_differences(initial_dice_scores, distilled_dice_scores, test_images_paths, f'{VISUALIZATIONS_ROOT_PATH}/{DATASET_NAME}_top_50_dice_differences.txt', k=50)
    #write_top_k_differences(initial_jaccard_scores, distilled_jaccard_scores, test_images_paths, f'{VISUALIZATIONS_ROOT_PATH}/{DATASET_NAME}_top_50_jaccard_differences.txt', k=50)

    # Save visualizations of the overlay between the ground truth mask and the mask predicted by the distilled model for all images in the test set
    original_images, gt_masks = load_images_and_masks(test_images_paths, test_masks_paths, HYPERPARAMETERS['image_size'])
    #save_overlay_visualization(test_images_paths, original_images, gt_masks, distilled_pred_masks, distilled_dice_scores, distilled_jaccard_scores, f'{VISUALIZATIONS_ROOT_PATH}/{DATASET_NAME}_overlays')
    
    # Save visualizations of the top 50 images with the largest absolute differences in Dice and Jaccard scores between the two models, showing the original image, the ground truth mask, the mask predicted by the dataset-specific model and the mask predicted by the distilled model
    save_top_differences(test_images_paths, original_images, gt_masks, initial_pred_masks, distilled_pred_masks, initial_dice_scores, distilled_dice_scores, initial_jaccard_scores, distilled_jaccard_scores, f'{VISUALIZATIONS_ROOT_PATH}/{DATASET_NAME}_top_differences', k=150)"""