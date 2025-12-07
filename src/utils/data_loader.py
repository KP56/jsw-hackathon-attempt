"""
Data loading utilities for tensor datasets.
"""
import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path


class TensorDataset(Dataset):
    """Dataset for tensor data with labels"""
    
    def __init__(self, data, labels):
        """
        Initialize dataset.
        
        Args:
            data (torch.Tensor): Input data
            labels (torch.Tensor): Labels
        """
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def load_tensors(tensors_dir, target_size=(540, 960)):
    """
    Load all .pt tensor files from a directory and combine them.
    
    Args:
        tensors_dir (str): Directory containing .pt tensor files
        target_size (tuple): Target size (height, width) for resizing
        
    Returns:
        tuple: (data, labels, seg_masks) tensors
    """
    tensors_path = Path(tensors_dir)
    
    # Find all .pt files in the directory
    tensor_files = sorted(list(tensors_path.glob("*.pt")))
    
    if not tensor_files:
        raise FileNotFoundError(f"No .pt tensor files found in {tensors_dir}")
    
    print(f"Found {len(tensor_files)} tensor file(s) in {tensors_dir}")
    
    all_data = []
    all_labels = []
    all_seg_masks = []
    
    for tensor_file in tensor_files:
        print(f"Loading: {tensor_file.name}")
        loaded = torch.load(tensor_file, map_location='cpu')
        
        # Expected format: (data, labels, seg_masks)
        if isinstance(loaded, (tuple, list)) and len(loaded) == 3:
            data, labels, seg_masks = loaded
            
            # Resize data (images) to target size
            if data.dim() == 4:  # (N, C, H, W)
                data_resized = F.interpolate(
                    data, 
                    size=target_size, 
                    mode='bilinear', 
                    align_corners=False
                )
            else:
                data_resized = data
            
            # Resize segmentation masks to target size
            if seg_masks.dim() == 3:  # (N, H, W)
                seg_masks_resized = F.interpolate(
                    seg_masks.unsqueeze(1),  # (N, 1, H, W)
                    size=target_size,
                    mode='bilinear',
                    align_corners=False
                ).squeeze(1)  # Back to (N, H, W)
            elif seg_masks.dim() == 4:  # Already (N, 1, H, W)
                seg_masks_resized = F.interpolate(
                    seg_masks,
                    size=target_size,
                    mode='bilinear',
                    align_corners=False
                ).squeeze(1)
            else:
                seg_masks_resized = seg_masks
            
            all_data.append(data_resized)
            all_labels.append(labels)
            all_seg_masks.append(seg_masks_resized)
            
            print(f"  Data shape: {data_resized.shape}, Labels: {labels.shape}, Masks: {seg_masks_resized.shape}")
        else:
            print(f"  Warning: Skipping {tensor_file.name} - unexpected format")
            continue
    
    if not all_data:
        raise ValueError("No valid tensor data loaded")
    
    # Concatenate all tensors
    combined_data = torch.cat(all_data, dim=0)
    combined_labels = torch.cat(all_labels, dim=0)
    combined_seg_masks = torch.cat(all_seg_masks, dim=0)
    
    print(f"\nCombined dataset:")
    print(f"  Data shape: {combined_data.shape}")
    print(f"  Labels shape: {combined_labels.shape}")
    print(f"  Segmentation masks shape: {combined_seg_masks.shape}")
    
    return combined_data, combined_labels, combined_seg_masks


def create_dataloaders(data, labels, seg_masks, config):
    """
    Create dataloaders for classification and segmentation tasks.
    
    Args:
        data (torch.Tensor): Input images
        labels (torch.Tensor): Classification labels
        seg_masks (torch.Tensor): Segmentation masks
        config (dict): Configuration dictionary
        
    Returns:
        dict: Dictionary containing train/val dataloaders for both tasks
    """
    # Extract config parameters
    train_split = config['data']['train_split']
    batch_size_cls = config['data']['batch_size_classifier']
    batch_size_seg = config['data']['batch_size_segmentation']
    num_workers = config['data']['num_workers']
    random_seed = config['data']['random_seed']
    
    # Create classification dataset
    cls_dataset = TensorDataset(data, labels)
    train_size = int(train_split * len(cls_dataset))
    val_size = len(cls_dataset) - train_size
    
    cls_train_dataset, cls_val_dataset = random_split(
        cls_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(random_seed)
    )
    
    cls_train_loader = DataLoader(
        cls_train_dataset,
        batch_size=batch_size_cls,
        shuffle=True,
        num_workers=num_workers
    )
    
    cls_val_loader = DataLoader(
        cls_val_dataset,
        batch_size=batch_size_cls,
        shuffle=False,
        num_workers=num_workers
    )
    
    print(f"\nClassification dataloaders:")
    print(f"  Train: {len(cls_train_dataset)} samples, {len(cls_train_loader)} batches")
    print(f"  Val: {len(cls_val_dataset)} samples, {len(cls_val_loader)} batches")
    
    # Create segmentation dataset
    seg_dataset = TensorDataset(data, seg_masks)
    seg_train_dataset, seg_val_dataset = random_split(
        seg_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(random_seed)
    )
    
    seg_train_loader = DataLoader(
        seg_train_dataset,
        batch_size=batch_size_seg,
        shuffle=True,
        num_workers=num_workers
    )
    
    seg_val_loader = DataLoader(
        seg_val_dataset,
        batch_size=batch_size_seg,
        shuffle=False,
        num_workers=num_workers
    )
    
    print(f"\nSegmentation dataloaders:")
    print(f"  Train: {len(seg_train_dataset)} samples, {len(seg_train_loader)} batches")
    print(f"  Val: {len(seg_val_dataset)} samples, {len(seg_val_loader)} batches")
    
    return {
        'classifier': {
            'train': cls_train_loader,
            'val': cls_val_loader
        },
        'segmentation': {
            'train': seg_train_loader,
            'val': seg_val_loader
        }
    }

