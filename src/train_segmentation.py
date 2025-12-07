"""
Training script for the segmentation model.
"""
import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models import SegmentationModel
from src.utils import load_tensors, create_dataloaders, calculate_iou, calculate_dice


def train_segmentation(model, train_loader, val_loader, config, device):
    """
    Train the segmentation model.
    
    Args:
        model: The segmentation model
        train_loader: Training data loader
        val_loader: Validation data loader
        config (dict): Configuration dictionary
        device: Device to train on
        
    Returns:
        dict: Training history
    """
    # Extract config
    num_epochs = config['segmentation']['num_epochs']
    lr = config['segmentation']['learning_rate']
    adam_betas = tuple(config['segmentation']['adam_betas'])
    checkpoint_dir = config['segmentation']['checkpoint_dir']
    
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=adam_betas)
    
    # History tracking
    history = {
        'train_loss': [],
        'val_loss': [],
        'iou': [],
        'dice': []
    }
    
    print("\n" + "="*70)
    print("TRAINING SEGMENTATION MODEL")
    print("="*70)
    print(f"Device: {device}")
    print(f"Epochs: {num_epochs}")
    print(f"Learning rate: {lr}")
    print(f"Adam betas: {adam_betas}")
    print("="*70)
    
    best_iou = 0.0
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for data, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            data, masks = data.to(device), masks.to(device).float()
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        iou_scores = []
        dice_scores = []
        
        with torch.no_grad():
            for data, masks in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]  "):
                data, masks = data.to(device), masks.to(device).float()
                outputs = model(data)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                
                # Calculate IoU and Dice for each sample in batch
                for i in range(outputs.shape[0]):
                    pred = (outputs[i] > 0.0).float()
                    target = masks[i]
                    
                    iou = calculate_iou(pred, target)
                    dice = calculate_dice(pred, target)
                    
                    if not np.isnan(iou):
                        iou_scores.append(iou)
                    if not np.isnan(dice):
                        dice_scores.append(dice)
        
        val_loss /= len(val_loader)
        mean_iou = np.mean(iou_scores) if iou_scores else 0.0
        mean_dice = np.mean(dice_scores) if dice_scores else 0.0
        
        # Store in history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['iou'].append(mean_iou)
        history['dice'].append(mean_dice)
        
        # Print metrics
        print(f"\nEpoch [{epoch+1}/{num_epochs}]")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Val IoU: {mean_iou:.4f}")
        print(f"  Val Dice: {mean_dice:.4f}")
        print("-"*70)
        
        # Save best model
        if mean_iou > best_iou:
            best_iou = mean_iou
            checkpoint_path = os.path.join(checkpoint_dir, 'best_segmentation.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'iou': mean_iou,
                'dice': mean_dice,
                'history': history
            }, checkpoint_path)
            print(f"  ✓ Saved best model (IoU: {mean_iou:.4f})")
    
    # Save final model
    final_checkpoint_path = os.path.join(checkpoint_dir, 'final_segmentation.pth')
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history
    }, final_checkpoint_path)
    
    print("\nSegmentation training complete!")
    print(f"Best IoU: {best_iou:.4f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    
    return history


def plot_training_history(history, save_path=None):
    """Plot training history"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Plot Loss
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', marker='o')
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', marker='s')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Segmentation: Training & Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot IoU and Dice
    axes[1].plot(epochs, history['iou'], 'purple', label='IoU', marker='o')
    axes[1].plot(epochs, history['dice'], 'orange', label='Dice', marker='s')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Score', fontsize=12)
    axes[1].set_title('Segmentation: IoU & Dice Scores', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved training history plot to {save_path}")
    
    plt.show()


def main():
    """Main training function"""
    # Load configuration
    config_path = Path(__file__).parent.parent / 'config.yml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set device
    if config['device']['use_cuda'] and torch.cuda.is_available():
        device = torch.device(f"cuda:{config['device']['cuda_device']}")
    else:
        device = torch.device('cpu')
    
    print(f"Using device: {device}")
    
    # Load data
    print("\n" + "="*70)
    print("LOADING DATA")
    print("="*70)
    
    tensors_dir = config['data']['tensors_dir']
    target_size = tuple(config['data']['target_size'])
    
    data, labels, seg_masks = load_tensors(tensors_dir, target_size)
    
    # Create dataloaders
    dataloaders = create_dataloaders(data, labels, seg_masks, config)
    train_loader = dataloaders['segmentation']['train']
    val_loader = dataloaders['segmentation']['val']
    
    # Create model
    print("\n" + "="*70)
    print("CREATING MODEL")
    print("="*70)
    
    in_channels = data.shape[1]
    hidden_size = config['model']['hidden_size']
    
    model = SegmentationModel(in_channels=in_channels, hidden_size=hidden_size).to(device)
    
    print(f"Segmentation model created:")
    print(f"  Input channels: {in_channels}")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    history = train_segmentation(model, train_loader, val_loader, config, device)
    
    # Plot training history
    plot_save_path = Path(config['segmentation']['checkpoint_dir']) / 'training_history.png'
    plot_training_history(history, save_path=plot_save_path)
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()

