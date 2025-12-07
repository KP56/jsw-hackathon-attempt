import torch
import torch.nn as nn
import numpy as np
import cv2


class Classifier(nn.Module):
    """CNN-based Image Classifier with Hough Transform channel"""
    
    def __init__(self, in_channels=3, hidden_size=8):
        super().__init__()

        # We will add 1 extra Hough channel → 4 channels total
        in_channels = 4

        # Convolutional layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_size, hidden_size * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_size * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_size * 2, hidden_size * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_size * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(hidden_size * 2, hidden_size * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_size * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_size * 4, hidden_size * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_size * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(hidden_size * 4, hidden_size * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_size * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_size * 8, hidden_size * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_size * 8),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_size * 8 * 4 * 4, hidden_size * 4),
            nn.BatchNorm1d(hidden_size * 4),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, 1)
        )


    # === SINGLE-IMAGE HOUGH ===
    @staticmethod
    def compute_hough_map(img_tensor):
        """
        img_tensor: (3, H, W)
        returns: (1, H, W)
        """
        img = img_tensor.detach().cpu().numpy()
        img = (img * 255).astype(np.uint8)

        gray = cv2.cvtColor(np.transpose(img, (1, 2, 0)), cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=50,
            minLineLength=30,
            maxLineGap=10
        )

        hough_map = np.zeros_like(gray, dtype=np.float32)

        if lines is not None:
            for x1, y1, x2, y2 in lines[:, 0]:
                cv2.line(hough_map, (x1, y1), (x2, y2), 255.0, 2)

        hough_map /= 255.0

        return torch.from_numpy(hough_map).unsqueeze(0)  # (1,H,W)


    # === BATCH FORWARD ===
    def forward(self, x):
        """
        x: (B, 3, H, W)
        """
        B, C, H, W = x.shape

        hough_maps = []

        # Process each image independently
        for i in range(B):
            hough_map = self.compute_hough_map(x[i])
            hough_maps.append(hough_map)

        # Stack to (B, 1, H, W)
        hough_maps = torch.stack(hough_maps).to(x.device)

        # Concatenate → (B, 4, H, W)
        x = torch.cat([x, hough_maps], dim=1)

        # Regular CNN forward
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.fc(x)

        return x.squeeze(-1)

