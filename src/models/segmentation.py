"""
DeepLabV3+ Segmentation Model for binary segmentation.
Uses pretrained MobileNetV3 backbone.
"""
import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large


class SegmentationModel(nn.Module):
    """DeepLabV3+ with MobileNetV3 backbone for binary segmentation"""
    
    def __init__(self, in_channels=3, hidden_size=8, pretrained=True):
        """
        Initialize the segmentation model.
        
        Args:
            in_channels (int): Number of input channels (3 for RGB)
            hidden_size (int): Not used, kept for compatibility
            pretrained (bool): Whether to use pretrained weights
        """
        super().__init__()
        
        # Load pretrained DeepLabV3+ with MobileNetV3 backbone
        # Note: We directly inherit from the model rather than wrapping it
        # This allows us to load checkpoints saved from training.ipynb
        base_model = deeplabv3_mobilenet_v3_large(pretrained=pretrained, progress=True)
        
        # Copy all attributes from base model
        self.backbone = base_model.backbone
        self.classifier = base_model.classifier
        self.aux_classifier = base_model.aux_classifier
        
        # Modify classifier for binary segmentation (1 output channel)
        self.classifier[4] = nn.Conv2d(256, 1, kernel_size=1)
        self.aux_classifier[4] = nn.Conv2d(10, 1, kernel_size=1)
        
        print(f"Loaded DeepLabV3+ with MobileNetV3 backbone")
        print(f"   Pretrained: {pretrained}")
        print(f"   Modified for binary segmentation (1 output channel)")

    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, H, W)
            
        Returns:
            torch.Tensor: Output logits of shape (batch_size, H, W)
        """
        # Forward through backbone
        features = self.backbone(x)
        
        # Forward through classifier (main output)
        output = self.classifier(features['out'])
        
        # Return dict to match DeepLabV3 interface
        result = {'out': output}
        
        # During training, also compute auxiliary output
        if self.training and 'aux' in features:
            result['aux'] = self.aux_classifier(features['aux'])
        
        # For inference, just return the main output
        if not self.training:
            output = result['out']
            # Squeeze channel dimension (from [B, 1, H, W] to [B, H, W])
            return output.squeeze(1)
        
        return result

