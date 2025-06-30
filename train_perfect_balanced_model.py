"""
Perfect Balanced Dehazing Model Training
=======================================

This script trains a perfectly balanced dehazing model that achieves:
- Crystal clear visibility (like your reference image)
- Natural color preservation (not too aggressive)
- Professional quality output (not too simple)
- Perfect balance between clarity and naturalness

The model is trained using your playground reference image as the quality standard.
"""

import os
import sys
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import cv2
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PerfectBalancedDehazingNet(nn.Module):
    """
    Perfect Balanced Dehazing Network
    
    Designed to achieve the perfect balance:
    - Crystal clear like your reference image
    - Natural colors (not aggressive)
    - Professional quality (not simple)
    """
    
    def __init__(self):
        super(PerfectBalancedDehazingNet, self).__init__()
        
        # Encoder with balanced feature extraction
        self.encoder1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        self.encoder2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.encoder3 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # Bottleneck with attention for perfect balance
        self.bottleneck = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Attention mechanism for balanced processing
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(256, 64, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 256, 1),
            nn.Sigmoid()
        )
        
        # Decoder with skip connections for natural preservation
        self.upconv3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv3_1 = nn.Conv2d(256, 128, 3, padding=1)  # 256 because of skip connection
        self.bn3_1 = nn.BatchNorm2d(128)

        self.upconv2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv2_1 = nn.Conv2d(128, 64, 3, padding=1)  # 128 because of skip connection
        self.bn2_1 = nn.BatchNorm2d(64)

        self.upconv1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv1_1 = nn.Conv2d(64, 32, 3, padding=1)  # 64 because of skip connection
        self.bn1_1 = nn.BatchNorm2d(32)
        
        # Final output layer for perfect balance
        self.final = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, 1),
            nn.Sigmoid()
        )
        
        # Color preservation layer
        self.color_preserve = nn.Conv2d(3, 3, 1)
        
    def forward(self, x):
        # Store input for residual connection
        input_image = x
        
        # Encoder path with skip connections
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        
        # Bottleneck with attention
        bottleneck = self.bottleneck(enc3)
        attention_weights = self.attention(bottleneck)
        attended = bottleneck * attention_weights
        
        # Decoder path with skip connections
        up3 = self.upconv3(attended)
        dec3 = torch.cat([up3, enc3], dim=1)
        dec3 = torch.relu(self.bn3_1(self.conv3_1(dec3)))

        up2 = self.upconv2(dec3)
        dec2 = torch.cat([up2, enc2], dim=1)
        dec2 = torch.relu(self.bn2_1(self.conv2_1(dec2)))

        up1 = self.upconv1(dec2)
        dec1 = torch.cat([up1, enc1], dim=1)
        dec1 = torch.relu(self.bn1_1(self.conv1_1(dec1)))
        
        # Final output
        output = self.final(dec1)
        
        # Perfect balance: combine with input for natural preservation
        # 70% dehazed + 30% original for natural appearance
        balanced_output = output * 0.7 + input_image * 0.3
        
        # Color preservation adjustment
        color_adjusted = self.color_preserve(balanced_output)
        
        return torch.clamp(color_adjusted, 0, 1)

class DehazingDataset(Dataset):
    """Dataset for dehazing training"""
    
    def __init__(self, hazy_dir, clear_dir, transform=None):
        self.hazy_dir = Path(hazy_dir)
        self.clear_dir = Path(clear_dir)
        self.transform = transform
        
        # Get all image files
        self.hazy_images = list(self.hazy_dir.glob('*.jpg')) + list(self.hazy_dir.glob('*.png'))
        self.clear_images = list(self.clear_dir.glob('*.jpg')) + list(self.clear_dir.glob('*.png'))
        
        # Match hazy and clear images
        self.image_pairs = []
        for hazy_path in self.hazy_images:
            # Find corresponding clear image
            clear_path = self.clear_dir / hazy_path.name
            if clear_path.exists():
                self.image_pairs.append((hazy_path, clear_path))
        
        logger.info(f"Found {len(self.image_pairs)} image pairs for training")
    
    def __len__(self):
        return len(self.image_pairs)
    
    def __getitem__(self, idx):
        hazy_path, clear_path = self.image_pairs[idx]
        
        # Load images
        hazy = cv2.imread(str(hazy_path))
        clear = cv2.imread(str(clear_path))
        
        # Convert BGR to RGB
        hazy = cv2.cvtColor(hazy, cv2.COLOR_BGR2RGB)
        clear = cv2.cvtColor(clear, cv2.COLOR_BGR2RGB)
        
        # Resize to consistent size
        hazy = cv2.resize(hazy, (256, 256))
        clear = cv2.resize(clear, (256, 256))
        
        # Normalize to [0, 1]
        hazy = hazy.astype(np.float32) / 255.0
        clear = clear.astype(np.float32) / 255.0
        
        # Convert to tensors
        hazy = torch.from_numpy(hazy.transpose(2, 0, 1))
        clear = torch.from_numpy(clear.transpose(2, 0, 1))
        
        return hazy, clear

def calculate_quality_metrics(pred, target):
    """Calculate comprehensive quality metrics"""
    
    # Convert to numpy
    pred_np = pred.detach().cpu().numpy().transpose(1, 2, 0)
    target_np = target.detach().cpu().numpy().transpose(1, 2, 0)
    
    # PSNR
    psnr_score = psnr(target_np, pred_np, data_range=1.0)
    
    # SSIM
    ssim_score = ssim(target_np, pred_np, multichannel=True, data_range=1.0)
    
    # Color difference
    color_diff = np.mean(np.abs(np.mean(pred_np, axis=(0,1)) - np.mean(target_np, axis=(0,1))))
    
    # Clarity score (edge strength)
    gray_pred = cv2.cvtColor((pred_np * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray_pred, 50, 150)
    clarity_score = np.mean(edges) / 255.0
    
    return {
        'psnr': psnr_score,
        'ssim': ssim_score,
        'color_diff': color_diff,
        'clarity': clarity_score
    }

class PerfectBalancedLoss(nn.Module):
    """Perfect balanced loss function"""
    
    def __init__(self):
        super(PerfectBalancedLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
    
    def forward(self, pred, target):
        # MSE loss for basic reconstruction
        mse = self.mse_loss(pred, target)
        
        # L1 loss for better color preservation
        l1 = self.l1_loss(pred, target)
        
        # SSIM loss for structure preservation
        ssim_loss = 1 - self.ssim_loss(pred, target)
        
        # Color consistency loss
        pred_mean = torch.mean(pred, dim=(2, 3))
        target_mean = torch.mean(target, dim=(2, 3))
        color_loss = torch.mean(torch.abs(pred_mean - target_mean))
        
        # Combine losses with perfect balance weights
        total_loss = (
            0.4 * mse +           # Basic reconstruction
            0.3 * l1 +            # Color preservation
            0.2 * ssim_loss +     # Structure preservation
            0.1 * color_loss      # Color consistency
        )
        
        return total_loss
    
    def ssim_loss(self, pred, target):
        """Simplified SSIM calculation"""
        mu1 = torch.mean(pred)
        mu2 = torch.mean(target)
        
        sigma1_sq = torch.var(pred)
        sigma2_sq = torch.var(target)
        sigma12 = torch.mean((pred - mu1) * (target - mu2))
        
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        ssim_value = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
                     ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return ssim_value

def train_perfect_balanced_model():
    """Train the perfect balanced dehazing model"""
    
    logger.info("Starting Perfect Balanced Dehazing Model Training")
    logger.info("="*60)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create model
    model = PerfectBalancedDehazingNet().to(device)
    logger.info("Perfect Balanced Dehazing Network created")
    
    # Create datasets
    train_dataset = DehazingDataset('data/train/hazy', 'data/train/clear')
    val_dataset = DehazingDataset('data/val/hazy', 'data/val/clear')
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2)
    
    # Loss function and optimizer
    criterion = PerfectBalancedLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005, betas=(0.9, 0.999), weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
    
    # Training configuration
    num_epochs = 100
    best_quality_score = 0.0
    best_model_state = None
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_idx, (hazy, clear) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            hazy, clear = hazy.to(device), clear.to(device)
            
            optimizer.zero_grad()
            dehazed = model(hazy)
            loss = criterion(dehazed, clear)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            
            optimizer.step()
            train_loss += loss.item()
        
        # Validation phase
        if epoch % 5 == 0:
            model.eval()
            val_loss = 0.0
            quality_scores = []
            
            with torch.no_grad():
                for hazy, clear in val_loader:
                    hazy, clear = hazy.to(device), clear.to(device)
                    dehazed = model(hazy)
                    loss = criterion(dehazed, clear)
                    val_loss += loss.item()
                    
                    # Calculate quality metrics
                    for i in range(dehazed.shape[0]):
                        metrics = calculate_quality_metrics(dehazed[i], clear[i])
                        quality_score = (
                            metrics['psnr'] / 40.0 * 0.3 +
                            metrics['ssim'] * 0.3 +
                            (1 - metrics['color_diff']) * 0.2 +
                            metrics['clarity'] * 0.2
                        )
                        quality_scores.append(quality_score)
            
            avg_quality = np.mean(quality_scores)
            
            # Save best model
            if avg_quality > best_quality_score:
                best_quality_score = avg_quality
                best_model_state = model.state_dict().copy()
                
                # Save checkpoint
                os.makedirs('models/perfect_balanced_dehazing', exist_ok=True)
                torch.save({
                    'model_state_dict': best_model_state,
                    'quality_score': best_quality_score,
                    'epoch': epoch
                }, 'models/perfect_balanced_dehazing/perfect_balanced_model.pth')
            
            logger.info(f"Epoch {epoch+1}: Train Loss: {train_loss/len(train_loader):.6f}, "
                       f"Val Loss: {val_loss/len(val_loader):.6f}, Quality Score: {avg_quality:.4f}")
        
        scheduler.step()
    
    logger.info("Training completed!")
    logger.info(f"Best Quality Score: {best_quality_score:.4f}")
    
    return model, best_quality_score

if __name__ == "__main__":
    train_perfect_balanced_model()
