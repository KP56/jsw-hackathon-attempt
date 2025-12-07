"""
Utility functions for saving and converting model checkpoints.
"""
import torch
from typing import Dict, Any, Optional


def save_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: Optional[int] = None,
    loss: Optional[float] = None,
    **kwargs
) -> None:
    """
    Save a model checkpoint with optional training metadata.
    
    Args:
        model: PyTorch model to save
        checkpoint_path: Path where to save the checkpoint
        optimizer: Optional optimizer state to save
        epoch: Optional epoch number
        loss: Optional loss value
        **kwargs: Additional metadata to save
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
    }
    
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    
    if epoch is not None:
        checkpoint["epoch"] = epoch
    
    if loss is not None:
        checkpoint["loss"] = loss
    
    # Add any additional metadata
    checkpoint.update(kwargs)
    
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to: {checkpoint_path}")


def save_state_dict(model: torch.nn.Module, state_dict_path: str) -> None:
    """
    Save only the model state_dict (simple format).
    
    Args:
        model: PyTorch model to save
        state_dict_path: Path where to save the state_dict
    """
    torch.save(model.state_dict(), state_dict_path)
    print(f"State dict saved to: {state_dict_path}")


def convert_checkpoint_to_state_dict(checkpoint_path: str, output_path: str) -> None:
    """
    Convert a checkpoint file to a simple state_dict file.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        output_path: Path where to save the state_dict
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        torch.save(state_dict, output_path)
        print(f"✓ Converted checkpoint to state_dict")
        print(f"  From: {checkpoint_path}")
        print(f"  To:   {output_path}")
        
        # Print metadata if available
        if "epoch" in checkpoint:
            print(f"  Epoch: {checkpoint['epoch']}")
        if "loss" in checkpoint:
            print(f"  Loss: {checkpoint['loss']:.4f}")
    
    elif isinstance(checkpoint, dict):
        # Already a state_dict
        torch.save(checkpoint, output_path)
        print(f"✓ File was already a state_dict, copied to: {output_path}")
    
    else:
        raise ValueError(f"Unrecognized checkpoint format in {checkpoint_path}")


def convert_state_dict_to_checkpoint(
    state_dict_path: str,
    output_path: str,
    epoch: Optional[int] = None,
    loss: Optional[float] = None
) -> None:
    """
    Convert a simple state_dict to a checkpoint file.
    
    Args:
        state_dict_path: Path to the state_dict file
        output_path: Path where to save the checkpoint
        epoch: Optional epoch number to include
        loss: Optional loss value to include
    """
    state_dict = torch.load(state_dict_path, map_location="cpu")
    
    checkpoint = {
        "model_state_dict": state_dict
    }
    
    if epoch is not None:
        checkpoint["epoch"] = epoch
    
    if loss is not None:
        checkpoint["loss"] = loss
    
    torch.save(checkpoint, output_path)
    print(f"✓ Converted state_dict to checkpoint")
    print(f"  From: {state_dict_path}")
    print(f"  To:   {output_path}")


def inspect_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    """
    Inspect a checkpoint or state_dict file and print its contents.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        
    Returns:
        Dictionary with inspection results
    """
    print(f"Inspecting: {checkpoint_path}")
    print("=" * 60)
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    info = {
        "path": checkpoint_path,
        "type": type(checkpoint).__name__,
    }
    
    if isinstance(checkpoint, dict):
        info["keys"] = list(checkpoint.keys())
        
        # Check if it's a checkpoint format
        if "model_state_dict" in checkpoint:
            print("Format: Checkpoint (with metadata)")
            print(f"Keys: {', '.join(checkpoint.keys())}")
            
            if "epoch" in checkpoint:
                print(f"Epoch: {checkpoint['epoch']}")
                info["epoch"] = checkpoint["epoch"]
            
            if "loss" in checkpoint:
                print(f"Loss: {checkpoint['loss']:.4f}")
                info["loss"] = checkpoint["loss"]
            
            state_dict = checkpoint["model_state_dict"]
        else:
            print("Format: Simple state_dict")
            state_dict = checkpoint
        
        # Analyze state_dict
        print(f"\nModel parameters: {len(state_dict)} tensors")
        info["num_parameters"] = len(state_dict)
        
        # Calculate total parameters
        total_params = sum(p.numel() for p in state_dict.values())
        print(f"Total parameters: {total_params:,}")
        info["total_parameters"] = total_params
        
        # Print first few parameter names and shapes
        print("\nFirst 5 parameters:")
        for i, (name, tensor) in enumerate(list(state_dict.items())[:5]):
            print(f"  {name}: {list(tensor.shape)}")
            if i == 0:
                info["sample_parameter"] = {
                    "name": name,
                    "shape": list(tensor.shape)
                }
    
    else:
        print(f"Warning: Unrecognized format (type: {type(checkpoint)})")
        info["warning"] = "Unrecognized format"
    
    print("=" * 60)
    
    return info


if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description="Checkpoint utility tools")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Inspect command
    inspect_parser = subparsers.add_parser("inspect", help="Inspect a checkpoint file")
    inspect_parser.add_argument("checkpoint_path", help="Path to checkpoint file")
    
    # Convert command
    convert_parser = subparsers.add_parser("convert", help="Convert between formats")
    convert_parser.add_argument("input_path", help="Input checkpoint/state_dict path")
    convert_parser.add_argument("output_path", help="Output path")
    convert_parser.add_argument("--to-state-dict", action="store_true",
                               help="Convert checkpoint to state_dict")
    convert_parser.add_argument("--to-checkpoint", action="store_true",
                               help="Convert state_dict to checkpoint")
    convert_parser.add_argument("--epoch", type=int, help="Epoch number (for checkpoint)")
    convert_parser.add_argument("--loss", type=float, help="Loss value (for checkpoint)")
    
    args = parser.parse_args()
    
    if args.command == "inspect":
        inspect_checkpoint(args.checkpoint_path)
    
    elif args.command == "convert":
        if args.to_state_dict:
            convert_checkpoint_to_state_dict(args.input_path, args.output_path)
        elif args.to_checkpoint:
            convert_state_dict_to_checkpoint(
                args.input_path, args.output_path,
                epoch=args.epoch, loss=args.loss
            )
        else:
            print("Error: Specify either --to-state-dict or --to-checkpoint")
            sys.exit(1)
    
    else:
        parser.print_help()

