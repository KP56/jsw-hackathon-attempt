"""
Visualization utilities for model predictions.
"""
import numpy as np
import matplotlib.pyplot as plt
import torch


def visualize_predictions(model, val_loader, device, model_type='classifier', num_samples=6, save_path=None):
    """
    Visualize model predictions.
    
    Args:
        model: The trained model
        val_loader: Validation data loader
        device: Device to run inference on
        model_type (str): 'classifier' or 'segmentation'
        num_samples (int): Number of samples to visualize
        save_path (str, optional): Path to save the figure
    """
    model.eval()
    
    if model_type == 'classifier':
        _visualize_classifier(model, val_loader, device, num_samples, save_path)
    elif model_type == 'segmentation':
        _visualize_segmentation(model, val_loader, device, num_samples, save_path)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def _visualize_classifier(model, val_loader, device, num_samples, save_path):
    """Visualize classifier predictions"""
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()
    
    samples_shown = 0
    
    with torch.no_grad():
        for data_batch, labels_batch in val_loader:
            data_batch = data_batch.to(device)
            outputs = model(data_batch)
            predictions = (outputs > 0.0).long()
            
            for i in range(len(data_batch)):
                if samples_shown >= num_samples:
                    break
                
                image = data_batch[i].cpu().numpy().transpose(1, 2, 0)
                image = (image - image.min()) / (image.max() - image.min() + 1e-8)
                
                true_label = labels_batch[i].item()
                pred_label = predictions[i].item()
                confidence = outputs[i].sigmoid().item()
                
                is_correct = (pred_label == true_label)
                color = 'green' if is_correct else 'red'
                
                axes[samples_shown].imshow(image)
                axes[samples_shown].set_title(
                    f"GT: {true_label} | Pred: {pred_label}\nConf: {confidence:.2f}",
                    color=color,
                    fontweight='bold'
                )
                axes[samples_shown].axis('off')
                
                samples_shown += 1
            
            if samples_shown >= num_samples:
                break
    
    plt.suptitle('Classifier Predictions (Green=Correct, Red=Incorrect)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    plt.show()


def _visualize_segmentation(model, val_loader, device, num_samples, save_path):
    """Visualize segmentation predictions"""
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    samples_shown = 0
    
    with torch.no_grad():
        for data_batch, masks_batch in val_loader:
            data_batch = data_batch.to(device)
            outputs = model(data_batch)
            predictions = (outputs > 0.0).float()
            
            for i in range(len(data_batch)):
                if samples_shown >= num_samples:
                    break
                
                image = data_batch[i].cpu().numpy().transpose(1, 2, 0)
                image = (image - image.min()) / (image.max() - image.min() + 1e-8)
                
                true_mask = masks_batch[i].cpu().numpy()
                pred_mask = predictions[i].cpu().numpy()
                
                # Image
                axes[samples_shown, 0].imshow(image)
                axes[samples_shown, 0].set_title('Input Image', fontweight='bold')
                axes[samples_shown, 0].axis('off')
                
                # Ground truth
                axes[samples_shown, 1].imshow(true_mask, cmap='gray', vmin=0, vmax=1)
                axes[samples_shown, 1].set_title('Ground Truth', fontweight='bold')
                axes[samples_shown, 1].axis('off')
                
                # Prediction
                axes[samples_shown, 2].imshow(pred_mask, cmap='gray', vmin=0, vmax=1)
                axes[samples_shown, 2].set_title('Prediction', fontweight='bold')
                axes[samples_shown, 2].axis('off')
                
                samples_shown += 1
            
            if samples_shown >= num_samples:
                break
    
    plt.suptitle('Segmentation Predictions', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    plt.show()

