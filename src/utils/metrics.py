"""
Metrics calculation utilities for segmentation tasks.
"""
import torch
import numpy as np


def _binarize_prediction(pred):
    """
    Convert prediction tensor to binary {0,1} tensor.
    - If values are outside [0,1], assume logits and threshold at 0 (logit > 0).
    - Otherwise assume probabilities and threshold at 0.5 (prob > 0.5).
    
    Args:
        pred (torch.Tensor): Prediction tensor
        
    Returns:
        torch.Tensor: Binary tensor with 0.0/1.0 values
    """
    pred = pred.clone()
    pred = pred.float()
    pmin = float(pred.min())
    pmax = float(pred.max())

    if pmin < 0.0 or pmax > 1.0:
        # Likely logits -> threshold at 0
        return (pred > 0.0).float()
    else:
        # Probabilities in [0,1] -> threshold at 0.5
        return (pred > 0.5).float()


def calculate_iou(pred, target, eps=1e-8):
    """
    Calculate IoU (Intersection over Union) for binary segmentation.
    
    Args:
        pred (torch.Tensor): Prediction tensor
        target (torch.Tensor): Target tensor (should be 0/1)
        eps (float): Small epsilon for numerical stability
        
    Returns:
        float: IoU score (or np.nan if union==0)
    """
    # Flatten
    pred = pred.view(-1)
    target = target.view(-1).float()

    pred_bin = _binarize_prediction(pred)
    target_bin = (target > 0.5).float()

    intersection = (pred_bin * target_bin).sum()
    union = (pred_bin + target_bin - pred_bin * target_bin).sum()

    if union.item() == 0:
        return float('nan')
    
    return (intersection / (union + eps)).item()


def calculate_dice(pred, target, eps=1e-8):
    """
    Calculate Dice coefficient (F1 score) for binary segmentation.
    
    Args:
        pred (torch.Tensor): Prediction tensor
        target (torch.Tensor): Target tensor (should be 0/1)
        eps (float): Small epsilon for numerical stability
        
    Returns:
        float: Dice coefficient
    """
    pred = pred.view(-1)
    target = target.view(-1).float()

    pred_bin = _binarize_prediction(pred)
    target_bin = (target > 0.5).float()

    intersection = (pred_bin * target_bin).sum()
    denom = pred_bin.sum() + target_bin.sum()

    if denom.item() == 0:
        # Both empty -> perfect match
        return 1.0
    
    return (2.0 * intersection / (denom + eps)).item()

