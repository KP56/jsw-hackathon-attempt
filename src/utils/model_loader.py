"""
Model loading utilities for the API.
"""
import os
import glob
import torch
from typing import Tuple, Union, Dict, Any

from src.models.classifier import Classifier
from src.models.segmentation import SegmentationModel


def load_state_dict_flexible(checkpoint_path: str, device: torch.device) -> Dict[str, Any]:
    """
    Load state dict from either a checkpoint dictionary or a simple state_dict file.
    
    This function handles both formats:
    1. Checkpoint format: {'model_state_dict': ..., 'optimizer_state_dict': ..., 'epoch': ..., etc.}
    2. Simple state_dict format: Direct OrderedDict of model parameters
    
    Args:
        checkpoint_path: Path to the checkpoint/state_dict file
        device: Device to load the checkpoint on
        
    Returns:
        Dictionary containing the model state_dict
        
    Raises:
        RuntimeError: If the file format is not recognized
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    
    # Load the file
    loaded_data = torch.load(checkpoint_path, map_location=device)
    
    # Check if it's a checkpoint dictionary with 'model_state_dict' key
    if isinstance(loaded_data, dict) and "model_state_dict" in loaded_data:
        print("  -> Detected checkpoint format (with 'model_state_dict' key)")
        state_dict = loaded_data["model_state_dict"]
        
        # Print additional info if available
        if "epoch" in loaded_data:
            print(f"  -> Checkpoint from epoch: {loaded_data['epoch']}")
        if "loss" in loaded_data:
            print(f"  -> Checkpoint loss: {loaded_data['loss']:.4f}")
            
        return state_dict
    
    # Check if it's a direct state_dict (OrderedDict or dict)
    elif isinstance(loaded_data, dict):
        # Verify it looks like a state_dict (has tensor values)
        sample_values = list(loaded_data.values())[:3]
        if sample_values and all(isinstance(v, torch.Tensor) for v in sample_values):
            print("  -> Detected simple state_dict format")
            return loaded_data
        else:
            raise RuntimeError(
                f"Loaded file from {checkpoint_path} is a dictionary but doesn't "
                "contain 'model_state_dict' key or valid tensor parameters. "
                f"Keys found: {list(loaded_data.keys())}"
            )
    
    else:
        raise RuntimeError(
            f"Unrecognized checkpoint format in {checkpoint_path}. "
            f"Expected either a checkpoint dict with 'model_state_dict' key "
            f"or a direct state_dict, but got {type(loaded_data)}"
        )


def load_models(config: dict, device: torch.device) -> Tuple[Classifier, SegmentationModel]:
    """
    Load classifier and segmentation models from checkpoints or state_dicts.
    
    Supports both formats:
    - Checkpoint format: {'model_state_dict': ..., 'optimizer_state_dict': ..., etc.}
    - Simple state_dict format: Direct model parameters
    
    Args:
        config: Configuration dictionary
        device: Device to load models on
        
    Returns:
        Tuple of (classifier_model, segmentation_model)
        
    Raises:
        FileNotFoundError: If checkpoint files are not found
    """
    print(f"Loading models on device: {device}")
    
    # Load classifier model
    classifier_model = Classifier(
        in_channels=config["model"]["in_channels"],
        hidden_size=config["model"]["hidden_size"]
    )
    
    # Load classifier checkpoint
    classifier_checkpoint_dir = config["classifier"]["checkpoint_dir"]
    classifier_checkpoints = glob.glob(os.path.join(classifier_checkpoint_dir, "*.pth"))
    
    if not classifier_checkpoints:
        raise FileNotFoundError(
            f"No classifier checkpoint found in {classifier_checkpoint_dir}!"
        )
    
    latest_classifier_checkpoint = max(classifier_checkpoints, key=os.path.getctime)
    print(f"Loading classifier from: {latest_classifier_checkpoint}")
    
    # Load state dict (handles both checkpoint and simple state_dict formats)
    classifier_state_dict = load_state_dict_flexible(latest_classifier_checkpoint, device)
    classifier_model.load_state_dict(classifier_state_dict)
    classifier_model = classifier_model.to(device)
    classifier_model.eval()
    print("+ Classifier loaded successfully!")
    
    # Load segmentation model
    segmentation_model = SegmentationModel(
        in_channels=config["model"]["in_channels"],
        hidden_size=config["model"]["hidden_size"]
    )
    
    # Load segmentation checkpoint
    segmentation_checkpoint_dir = config["segmentation"]["checkpoint_dir"]
    segmentation_checkpoints = glob.glob(os.path.join(segmentation_checkpoint_dir, "*.pth"))
    
    if not segmentation_checkpoints:
        raise FileNotFoundError(
            f"No segmentation checkpoint found in {segmentation_checkpoint_dir}!"
        )
    
    latest_segmentation_checkpoint = max(segmentation_checkpoints, key=os.path.getctime)
    print(f"Loading segmentation from: {latest_segmentation_checkpoint}")
    
    # Load state dict (handles both checkpoint and simple state_dict formats)
    segmentation_state_dict = load_state_dict_flexible(latest_segmentation_checkpoint, device)
    segmentation_model.load_state_dict(segmentation_state_dict)
    segmentation_model = segmentation_model.to(device)
    segmentation_model.eval()
    print("+ Segmentation model loaded successfully!")
    
    print("\n+ All models loaded successfully!")
    
    return classifier_model, segmentation_model


def get_device(config: dict) -> torch.device:
    """
    Get the appropriate device based on configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        torch.device for model inference
    """
    if config["device"]["use_cuda"] and torch.cuda.is_available():
        device = torch.device(f"cuda:{config['device']['cuda_device']}")
    else:
        device = torch.device("cpu")
    
    return device

