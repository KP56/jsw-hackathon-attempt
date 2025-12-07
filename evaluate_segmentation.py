#!/usr/bin/env python3
"""
Evaluate segmentation model using IoU and Dice Coefficient metrics.

This script:
1. Loads COCO format annotations from images/result.json
2. Generates ground truth masks from polygon annotations
3. Runs segmentation model on each image
4. Calculates IoU and Dice Coefficient for each image
5. Reports aggregate metrics and saves detailed results
"""
import os
import sys
import json
import numpy as np
import cv2
import torch
import yaml
from typing import Dict, List, Tuple, Any
from tqdm import tqdm
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.models.segmentation import SegmentationModel
from src.utils.model_loader import load_models, get_device
from src.utils.image_processing import (
    load_image,
    preprocess_image_for_segmentation,
    compute_mask,
    resize_mask_to_original
)


def load_config() -> dict:
    """Load configuration from config.yml"""
    with open("config.yml", "r") as f:
        config = yaml.safe_load(f)
    return config


def load_coco_annotations(json_path: str) -> Dict[str, Any]:
    """
    Load COCO format annotations.
    
    Args:
        json_path: Path to COCO JSON file
        
    Returns:
        Dictionary with images, annotations, and categories
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    print(f"Loaded COCO annotations:")
    print(f"  - Images: {len(data['images'])}")
    print(f"  - Annotations: {len(data['annotations'])}")
    print(f"  - Categories: {len(data['categories'])}")
    
    return data


def polygon_to_mask(polygon: List[float], height: int, width: int) -> np.ndarray:
    """
    Convert polygon coordinates to binary mask.
    
    Args:
        polygon: List of x, y coordinates [x1, y1, x2, y2, ...]
        height: Image height
        width: Image width
        
    Returns:
        Binary mask (height x width)
    """
    # Reshape polygon to (N, 2) array
    polygon_array = np.array(polygon).reshape(-1, 2)
    polygon_array = polygon_array.astype(np.int32)
    
    # Create empty mask
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Fill polygon
    cv2.fillPoly(mask, [polygon_array], 1)
    
    return mask


def rle_to_mask(rle: Dict, height: int, width: int) -> np.ndarray:
    """
    Convert RLE (Run Length Encoding) to binary mask.
    
    Args:
        rle: RLE dictionary with 'counts' and 'size'
        height: Image height
        width: Image width
        
    Returns:
        Binary mask (height x width)
    """
    from pycocotools import mask as coco_mask
    if isinstance(rle, list):
        # Multiple polygons - use polygon_to_mask
        mask = np.zeros((height, width), dtype=np.uint8)
        for polygon in rle:
            poly_mask = polygon_to_mask(polygon, height, width)
            mask = np.maximum(mask, poly_mask)
        return mask
    else:
        # RLE format
        rle['size'] = [height, width]
        return coco_mask.decode(rle)


def get_ground_truth_mask(annotation: Dict, height: int, width: int) -> np.ndarray:
    """
    Get ground truth mask from COCO annotation.
    
    Args:
        annotation: COCO annotation dictionary
        height: Image height
        width: Image width
        
    Returns:
        Binary mask (height x width)
    """
    segmentation = annotation['segmentation']
    
    if isinstance(segmentation, list):
        # Polygon format
        if len(segmentation) == 1:
            # Single polygon
            return polygon_to_mask(segmentation[0], height, width)
        else:
            # Multiple polygons - combine them
            mask = np.zeros((height, width), dtype=np.uint8)
            for polygon in segmentation:
                poly_mask = polygon_to_mask(polygon, height, width)
                mask = np.maximum(mask, poly_mask)
            return mask
    else:
        # RLE format
        return rle_to_mask(segmentation, height, width)


def calculate_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """
    Calculate Intersection over Union (IoU).
    
    IoU = |Intersection| / |Union|
    
    Args:
        pred_mask: Predicted binary mask (0 or 1)
        gt_mask: Ground truth binary mask (0 or 1)
        
    Returns:
        IoU score (0.0 to 1.0)
    """
    # Ensure binary masks
    pred_mask = (pred_mask > 0).astype(np.uint8)
    gt_mask = (gt_mask > 0).astype(np.uint8)
    
    # Calculate intersection and union
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    
    # Avoid division by zero
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    iou = intersection / union
    return float(iou)


def calculate_dice(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """
    Calculate Dice Coefficient (F1 Score).
    
    Dice = 2 * |Intersection| / (|A| + |B|)
    
    Args:
        pred_mask: Predicted binary mask (0 or 1)
        gt_mask: Ground truth binary mask (0 or 1)
        
    Returns:
        Dice coefficient (0.0 to 1.0)
    """
    # Ensure binary masks
    pred_mask = (pred_mask > 0).astype(np.uint8)
    gt_mask = (gt_mask > 0).astype(np.uint8)
    
    # Calculate intersection
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    
    # Calculate sum of both masks
    pred_sum = pred_mask.sum()
    gt_sum = gt_mask.sum()
    
    # Avoid division by zero
    if pred_sum + gt_sum == 0:
        return 1.0 if intersection == 0 else 0.0
    
    dice = (2.0 * intersection) / (pred_sum + gt_sum)
    return float(dice)


def calculate_precision_recall(pred_mask: np.ndarray, gt_mask: np.ndarray) -> Tuple[float, float]:
    """
    Calculate Precision and Recall.
    
    Args:
        pred_mask: Predicted binary mask
        gt_mask: Ground truth binary mask
        
    Returns:
        Tuple of (precision, recall)
    """
    pred_mask = (pred_mask > 0).astype(np.uint8)
    gt_mask = (gt_mask > 0).astype(np.uint8)
    
    true_positive = np.logical_and(pred_mask, gt_mask).sum()
    false_positive = np.logical_and(pred_mask, np.logical_not(gt_mask)).sum()
    false_negative = np.logical_and(np.logical_not(pred_mask), gt_mask).sum()
    
    # Precision: TP / (TP + FP)
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0.0
    
    # Recall: TP / (TP + FN)
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0.0
    
    return float(precision), float(recall)


def save_comparison_image(
    image: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    output_path: str,
    iou: float,
    dice: float
):
    """
    Save comparison visualization.
    
    Args:
        image: Original image
        gt_mask: Ground truth mask
        pred_mask: Predicted mask
        output_path: Path to save comparison
        iou: IoU score
        dice: Dice score
    """
    # Create visualization with 3 panels
    h, w = image.shape[:2]
    vis = np.zeros((h, w * 3, 3), dtype=np.uint8)
    
    # Panel 1: Original image with ground truth (green)
    panel1 = image.copy()
    panel1[gt_mask > 0] = [0, 255, 0]
    vis[:, :w] = panel1
    
    # Panel 2: Original image with prediction (blue)
    panel2 = image.copy()
    panel2[pred_mask > 0] = [255, 0, 0]
    vis[:, w:2*w] = panel2
    
    # Panel 3: Comparison (green=correct, red=FP, yellow=FN)
    panel3 = image.copy()
    tp = np.logical_and(gt_mask, pred_mask)
    fp = np.logical_and(pred_mask, np.logical_not(gt_mask))
    fn = np.logical_and(gt_mask, np.logical_not(pred_mask))
    
    panel3[tp > 0] = [0, 255, 0]      # Green: True Positive
    panel3[fp > 0] = [0, 0, 255]      # Red: False Positive
    panel3[fn > 0] = [0, 255, 255]    # Yellow: False Negative
    vis[:, 2*w:] = panel3
    
    # Add text labels
    cv2.putText(vis, "Ground Truth", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(vis, "Prediction", (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(vis, "Comparison", (2*w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Add metrics
    cv2.putText(vis, f"IoU: {iou:.4f}", (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(vis, f"Dice: {dice:.4f}", (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    cv2.imwrite(output_path, vis)


def evaluate_segmentation(
    images_dir: str = "./images",
    annotations_path: str = "./images/result.json",
    save_visualizations: bool = True,
    output_dir: str = "./evaluation_results"
):
    """
    Evaluate segmentation model on COCO annotations.
    
    Args:
        images_dir: Directory containing images
        annotations_path: Path to COCO JSON file
        save_visualizations: Whether to save comparison images
        output_dir: Directory to save results
    """
    print("=" * 80)
    print("Segmentation Model Evaluation")
    print("=" * 80)
    
    # Load configuration and model
    print("\n1. Loading configuration and model...")
    config = load_config()
    device = get_device(config)
    print(f"   Device: {device}")
    
    _, segmentation_model = load_models(config, device)
    segmentation_model.eval()
    
    target_size = tuple(config["data"]["target_size"])
    
    # Load COCO annotations
    print("\n2. Loading COCO annotations...")
    coco_data = load_coco_annotations(annotations_path)
    
    # Create image_id to annotation mapping
    image_id_to_ann = {}
    for ann in coco_data['annotations']:
        image_id_to_ann[ann['image_id']] = ann
    
    # Create output directories
    if save_visualizations:
        os.makedirs(output_dir, exist_ok=True)
        vis_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
    
    # Create directory for overlayed masks
    overlayed_dir = os.path.join("output", "overlayed_masks")
    os.makedirs(overlayed_dir, exist_ok=True)
    
    # Evaluate each image
    print("\n3. Evaluating images...")
    results = []
    
    for image_info in tqdm(coco_data['images'], desc="Processing"):
        image_id = image_info['id']
        file_name = image_info['file_name']
        
        # Handle path formatting
        if file_name.startswith("images/") or file_name.startswith("images\\"):
            file_name = file_name.replace("images/", "").replace("images\\", "")
        
        image_path = os.path.join(images_dir, file_name)
        
        # Check if annotation exists
        if image_id not in image_id_to_ann:
            print(f"Warning: No annotation for image {image_id} ({file_name})")
            continue
        
        annotation = image_id_to_ann[image_id]
        
        try:
            # Load image
            image = load_image(image_path)
            original_height, original_width = image.shape[:2]
            
            # Resize image to 960x540 before segmentation
            eval_size = (960, 540)  # (width, height)
            image_resized = cv2.resize(image, eval_size, interpolation=cv2.INTER_LINEAR)
            height, width = image_resized.shape[:2]
            
            # Get ground truth mask at original size first
            gt_mask_original = get_ground_truth_mask(annotation, original_height, original_width)
            # Resize ground truth mask to match resized image
            gt_mask = cv2.resize(gt_mask_original, eval_size, interpolation=cv2.INTER_NEAREST)
            
            # Run segmentation model on resized image
            image_tensor, original_size = preprocess_image_for_segmentation(image_resized, target_size)
            pred_mask_model = compute_mask(image_tensor, segmentation_model, device)
            pred_mask = resize_mask_to_original(pred_mask_model, original_size)
            
            # Calculate metrics
            iou = calculate_iou(pred_mask, gt_mask)
            dice = calculate_dice(pred_mask, gt_mask)
            precision, recall = calculate_precision_recall(pred_mask, gt_mask)
            
            # Store results
            result = {
                'image_id': image_id,
                'file_name': file_name,
                'iou': iou,
                'dice': dice,
                'precision': precision,
                'recall': recall,
                'gt_pixels': int(gt_mask.sum()),
                'pred_pixels': int(pred_mask.sum())
            }
            results.append(result)
            
            # Save visualization (using resized image)
            if save_visualizations and (len(results) <= 10 or iou < 0.5):  # Save first 10 or poor results
                vis_path = os.path.join(vis_dir, f"{os.path.splitext(file_name)[0]}_comparison.png")
                save_comparison_image(image_resized, gt_mask, pred_mask, vis_path, iou, dice)
            
            # Save overlayed mask (for all images, using resized image)
            overlay_path = os.path.join(overlayed_dir, f"{os.path.splitext(file_name)[0]}_overlay.png")
            # Create overlay with semi-transparent mask
            overlay_img = image_resized.copy()
            overlay_img[pred_mask > 0] = overlay_img[pred_mask > 0] * 0.5 + np.array([0, 255, 0]) * 0.5
            cv2.imwrite(overlay_path, overlay_img)
        
        except Exception as e:
            print(f"Error processing {file_name}: {e}")
            continue
    
    # Calculate aggregate metrics
    print("\n4. Calculating aggregate metrics...")
    ious = [r['iou'] for r in results]
    dices = [r['dice'] for r in results]
    precisions = [r['precision'] for r in results]
    recalls = [r['recall'] for r in results]
    
    mean_iou = np.mean(ious)
    mean_dice = np.mean(dices)
    mean_precision = np.mean(precisions)
    mean_recall = np.mean(recalls)
    
    std_iou = np.std(ious)
    std_dice = np.std(dices)
    
    # Print results
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(f"\nDataset: {len(results)} images")
    print(f"\nMean IoU (Intersection over Union): {mean_iou:.4f} ± {std_iou:.4f}")
    print(f"Mean Dice Coefficient (F1 Score):   {mean_dice:.4f} ± {std_dice:.4f}")
    print(f"Mean Precision:                      {mean_precision:.4f}")
    print(f"Mean Recall:                         {mean_recall:.4f}")
    
    print(f"\nIoU Distribution:")
    print(f"  Min:     {np.min(ious):.4f}")
    print(f"  25th %:  {np.percentile(ious, 25):.4f}")
    print(f"  Median:  {np.median(ious):.4f}")
    print(f"  75th %:  {np.percentile(ious, 75):.4f}")
    print(f"  Max:     {np.max(ious):.4f}")
    
    print(f"\nDice Distribution:")
    print(f"  Min:     {np.min(dices):.4f}")
    print(f"  25th %:  {np.percentile(dices, 25):.4f}")
    print(f"  Median:  {np.median(dices):.4f}")
    print(f"  75th %:  {np.percentile(dices, 75):.4f}")
    print(f"  Max:     {np.max(dices):.4f}")
    
    # Save detailed results
    results_file = os.path.join(output_dir, "evaluation_results.json")
    with open(results_file, 'w') as f:
        json.dump({
            'summary': {
                'num_images': len(results),
                'mean_iou': float(mean_iou),
                'std_iou': float(std_iou),
                'mean_dice': float(mean_dice),
                'std_dice': float(std_dice),
                'mean_precision': float(mean_precision),
                'mean_recall': float(mean_recall),
                'min_iou': float(np.min(ious)),
                'max_iou': float(np.max(ious)),
                'median_iou': float(np.median(ious)),
                'min_dice': float(np.min(dices)),
                'max_dice': float(np.max(dices)),
                'median_dice': float(np.median(dices))
            },
            'per_image_results': results
        }, f, indent=2)
    
    print(f"\nDetailed results saved to: {results_file}")
    if save_visualizations:
        print(f"Visualizations saved to: {vis_dir}")
    print(f"Overlayed masks saved to: {overlayed_dir}")
    
    print("=" * 80)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate segmentation model")
    parser.add_argument(
        "--images-dir",
        default="./images",
        help="Directory containing images (default: ./images)"
    )
    parser.add_argument(
        "--annotations",
        default="./images/result.json",
        help="Path to COCO JSON file (default: ./images/result.json)"
    )
    parser.add_argument(
        "--output-dir",
        default="./evaluation_results",
        help="Output directory for results (default: ./evaluation_results)"
    )
    parser.add_argument(
        "--no-visualizations",
        action="store_true",
        help="Don't save visualization images"
    )
    
    args = parser.parse_args()
    
    evaluate_segmentation(
        images_dir=args.images_dir,
        annotations_path=args.annotations,
        save_visualizations=not args.no_visualizations,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()

