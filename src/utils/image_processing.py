"""
Image processing utilities for mask computation.
"""
import os
import glob
import cv2
import numpy as np
import torch
from typing import List, Dict, Any, Tuple


def load_image(image_path: str) -> np.ndarray:
    """
    Load an image from file.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Image as numpy array in BGR format
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    return image


def preprocess_image_for_segmentation(
    image: np.ndarray,
    target_size: Tuple[int, int]
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Preprocess an image for segmentation model input.
    
    Args:
        image: BGR image from OpenCV
        target_size: (height, width) tuple for model input
        
    Returns:
        Tuple of (preprocessed tensor, original size)
    """
    original_size = (image.shape[0], image.shape[1])  # (height, width)
    
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize to target size
    image_resized = cv2.resize(image_rgb, (target_size[1], target_size[0]))
    
    # Convert to tensor and normalize to [0, 1]
    image_tensor = torch.from_numpy(image_resized).float() / 255.0
    
    # Change from HWC to CHW format
    image_tensor = image_tensor.permute(2, 0, 1)
    
    return image_tensor, original_size


def compute_mask(
    image_tensor: torch.Tensor,
    segmentation_model: torch.nn.Module,
    device: torch.device
) -> np.ndarray:
    """
    Compute segmentation mask for an image.
    
    Args:
        image_tensor: Preprocessed image tensor
        segmentation_model: Trained segmentation model
        device: Device to run inference on
        
    Returns:
        Binary segmentation mask (H x W)
    """
    with torch.no_grad():
        image_batch = image_tensor.unsqueeze(0).to(device)
        logits = segmentation_model(image_batch)
        probs = torch.sigmoid(logits)
        mask = (probs > 0.5).squeeze(0).cpu().numpy().astype(np.uint8)
    
    return mask


def resize_mask_to_original(
    mask: np.ndarray,
    original_size: Tuple[int, int]
) -> np.ndarray:
    """
    Resize mask back to original image size.
    
    Args:
        mask: Binary mask at model resolution
        original_size: (height, width) of original image
        
    Returns:
        Binary mask at original resolution
    """
    mask_resized = cv2.resize(
        mask,
        (original_size[1], original_size[0]),
        interpolation=cv2.INTER_NEAREST
    )
    return mask_resized


def save_mask(mask: np.ndarray, output_path: str) -> None:
    """
    Save binary mask to file.
    
    Args:
        mask: Binary mask (values 0 or 1)
        output_path: Path to save mask
    """
    # Convert to 0-255 range for saving
    mask_image = (mask * 255).astype(np.uint8)
    cv2.imwrite(output_path, mask_image)


def save_mask_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    output_path: str,
    alpha: float = 0.5,
    color: Tuple[int, int, int] = (0, 255, 0)
) -> None:
    """
    Save image with mask overlay.
    
    Args:
        image: Original BGR image
        mask: Binary mask (values 0 or 1)
        output_path: Path to save overlay image
        alpha: Transparency of overlay (0-1)
        color: BGR color for mask overlay
    """
    # Create colored mask
    overlay = image.copy()
    overlay[mask > 0] = color
    
    # Blend with original image
    result = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)
    
    cv2.imwrite(output_path, result)


def get_mask_statistics(mask: np.ndarray) -> Dict[str, Any]:
    """
    Compute statistics about a mask.
    
    Args:
        mask: Binary mask (values 0 or 1)
        
    Returns:
        Dictionary with mask statistics
    """
    total_pixels = mask.size
    masked_pixels = np.sum(mask > 0)
    coverage = (masked_pixels / total_pixels) * 100
    
    # Find bounding box
    rows, cols = np.where(mask > 0)
    if len(rows) > 0:
        min_row, max_row = int(np.min(rows)), int(np.max(rows))
        min_col, max_col = int(np.min(cols)), int(np.max(cols))
        bbox = {
            "min_y": min_row,
            "max_y": max_row,
            "min_x": min_col,
            "max_x": max_col,
            "width": max_col - min_col + 1,
            "height": max_row - min_row + 1
        }
    else:
        bbox = None
    
    return {
        "total_pixels": int(total_pixels),
        "masked_pixels": int(masked_pixels),
        "coverage_percent": float(coverage),
        "bounding_box": bbox
    }


def find_images(directory: str) -> List[str]:
    """
    Find all image files in a directory.
    
    Args:
        directory: Directory to search for images
        
    Returns:
        List of image file paths
    """
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif', '*.webp']
    image_files = []
    
    for ext in image_extensions:
        pattern = os.path.join(directory, ext)
        image_files.extend(glob.glob(pattern))
        # Also check uppercase extensions
        pattern_upper = os.path.join(directory, ext.upper())
        image_files.extend(glob.glob(pattern_upper))
    
    return sorted(image_files)

