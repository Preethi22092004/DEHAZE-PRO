"""
Reference-Guided Training System
===============================

This system uses your playground reference image to guide the training process,
ensuring the model learns to produce exactly the quality you want:
- Crystal clear like your reference
- Natural colors (not aggressive)
- Perfect balance (not too simple)
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

from train_perfect_balanced_model import PerfectBalancedDehazingNet, DehazingDataset

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ReferenceGuidedLoss(nn.Module):
    """Reference-guided loss function that matches your playground image quality"""
    
    def __init__(self, reference_image_path):
        super(ReferenceGuidedLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        
        # Load and process reference image
        self.reference_features = self.extract_reference_features(reference_image_path)
        
    def extract_reference_features(self, image_path):
        """Extract quality features from reference image"""
        if not os.path.exists(image_path):
            logger.warning(f"Reference image not found: {image_path}")
            return None
        
        # Load reference image
        ref_image = cv2.imread(image_path)
        ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)
        ref_image = cv2.resize(ref_image, (256, 256))
        ref_image = ref_image.astype(np.float32) / 255.0
        
        # Extract features
        features = {
            'brightness': np.mean(ref_image),
            'contrast': np.std(ref_image),
            'color_balance': np.mean(ref_image, axis=(0, 1)),
            'edge_strength': self.calculate_edge_strength(ref_image),
            'clarity_level': self.calculate_clarity_level(ref_image)
        }
        
        logger.info(f"Reference features extracted from {image_path}")
        logger.info(f"  Brightness: {features['brightness']:.3f}")
        logger.info(f"  Contrast: {features['contrast']:.3f}")
        logger.info(f"  Edge Strength: {features['edge_strength']:.3f}")
        logger.info(f"  Clarity Level: {features['clarity_level']:.3f}")
        
        return features
    
    def calculate_edge_strength(self, image):
        """Calculate edge strength of image"""
        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        return np.mean(edges) / 255.0
    
    def calculate_clarity_level(self, image):
        """Calculate clarity level of image"""
        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        
        # Laplacian variance for clarity
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        clarity = np.var(laplacian) / 10000.0  # Normalize
        
        return min(clarity, 1.0)
    
    def forward(self, pred, target):
        """Calculate reference-guided loss"""
        
        # Basic reconstruction losses
        mse = self.mse_loss(pred, target)
        l1 = self.l1_loss(pred, target)
        
        # SSIM loss for structure preservation
        ssim_loss = 1 - self.calculate_ssim(pred, target)
        
        # Color consistency loss
        pred_mean = torch.mean(pred, dim=(2, 3))
        target_mean = torch.mean(target, dim=(2, 3))
        color_loss = torch.mean(torch.abs(pred_mean - target_mean))
        
        # Reference-guided losses
        reference_loss = 0.0
        if self.reference_features:
            reference_loss = self.calculate_reference_loss(pred)
        
        # Perfect balance combination
        total_loss = (
            0.3 * mse +              # Basic reconstruction
            0.25 * l1 +              # Color preservation
            0.2 * ssim_loss +        # Structure preservation
            0.15 * color_loss +      # Color consistency
            0.1 * reference_loss     # Reference matching
        )
        
        return total_loss
    
    def calculate_ssim(self, pred, target):
        """Calculate SSIM between predicted and target"""
        # Convert to grayscale
        pred_gray = 0.299 * pred[:, 0] + 0.587 * pred[:, 1] + 0.114 * pred[:, 2]
        target_gray = 0.299 * target[:, 0] + 0.587 * target[:, 1] + 0.114 * target[:, 2]
        
        # Simplified SSIM calculation
        mu1 = torch.mean(pred_gray)
        mu2 = torch.mean(target_gray)
        
        sigma1_sq = torch.var(pred_gray)
        sigma2_sq = torch.var(target_gray)
        sigma12 = torch.mean((pred_gray - mu1) * (target_gray - mu2))
        
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        ssim_value = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
                     ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return torch.clamp(ssim_value, 0, 1)
    
    def calculate_reference_loss(self, pred):
        """Calculate loss based on reference image features"""
        
        # Convert prediction to numpy for feature extraction
        pred_np = pred.detach().cpu().numpy()
        
        total_ref_loss = 0.0
        batch_size = pred_np.shape[0]
        
        for i in range(batch_size):
            img = pred_np[i].transpose(1, 2, 0)
            
            # Extract features from prediction
            pred_brightness = np.mean(img)
            pred_contrast = np.std(img)
            pred_edge_strength = self.calculate_edge_strength(img)
            pred_clarity = self.calculate_clarity_level(img)
            
            # Compare with reference features
            brightness_diff = abs(pred_brightness - self.reference_features['brightness'])
            contrast_diff = abs(pred_contrast - self.reference_features['contrast'])
            edge_diff = abs(pred_edge_strength - self.reference_features['edge_strength'])
            clarity_diff = abs(pred_clarity - self.reference_features['clarity_level'])
            
            # Combine reference losses
            ref_loss = (
                brightness_diff * 0.3 +
                contrast_diff * 0.3 +
                edge_diff * 0.2 +
                clarity_diff * 0.2
            )
            
            total_ref_loss += ref_loss
        
        return torch.tensor(total_ref_loss / batch_size, device=pred.device, requires_grad=True)

class ReferenceGuidedTrainer:
    """Reference-guided trainer for perfect balanced dehazing"""
    
    def __init__(self, reference_image_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.reference_image_path = reference_image_path
        
        # Create model
        self.model = PerfectBalancedDehazingNet().to(self.device)
        
        # Create reference-guided loss
        self.criterion = ReferenceGuidedLoss(reference_image_path)
        
        # Optimizer with perfect balance settings
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=0.0003,  # Lower learning rate for stable training
            betas=(0.9, 0.999),
            weight_decay=1e-4
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=150, eta_min=1e-6
        )
        
        # Training state
        self.best_quality_score = 0.0
        self.best_model_state = None
        self.training_history = []
        
        logger.info("Reference-guided trainer initialized")
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for hazy, clear in tqdm(train_loader, desc="Training"):
            hazy, clear = hazy.to(self.device), clear.to(self.device)
            
            self.optimizer.zero_grad()
            dehazed = self.model(hazy)
            loss = self.criterion(dehazed, clear)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def validate_epoch(self, val_loader):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        quality_scores = []
        
        with torch.no_grad():
            for hazy, clear in val_loader:
                hazy, clear = hazy.to(self.device), clear.to(self.device)
                dehazed = self.model(hazy)
                loss = self.criterion(dehazed, clear)
                total_loss += loss.item()
                
                # Calculate quality metrics
                for i in range(dehazed.shape[0]):
                    quality = self.calculate_quality_score(dehazed[i], clear[i])
                    quality_scores.append(quality)
        
        avg_loss = total_loss / len(val_loader)
        avg_quality = np.mean(quality_scores) if quality_scores else 0.0
        
        return avg_loss, avg_quality
    
    def calculate_quality_score(self, pred, target):
        """Calculate comprehensive quality score"""
        
        # Convert to numpy
        pred_np = pred.detach().cpu().numpy().transpose(1, 2, 0)
        target_np = target.detach().cpu().numpy().transpose(1, 2, 0)
        
        # PSNR
        psnr_score = psnr(target_np, pred_np, data_range=1.0)
        
        # SSIM with appropriate window size
        min_dim = min(target_np.shape[0], target_np.shape[1])
        win_size = min(7, min_dim) if min_dim >= 7 else 3
        if win_size % 2 == 0:
            win_size -= 1
        ssim_score = ssim(target_np, pred_np, multichannel=True, data_range=1.0, win_size=win_size)
        
        # Color difference
        color_diff = np.mean(np.abs(np.mean(pred_np, axis=(0,1)) - np.mean(target_np, axis=(0,1))))
        
        # Clarity score
        gray_pred = cv2.cvtColor((pred_np * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray_pred, 50, 150)
        clarity_score = np.mean(edges) / 255.0
        
        # Reference similarity (if available)
        ref_similarity = 1.0
        if self.criterion.reference_features:
            pred_brightness = np.mean(pred_np)
            pred_contrast = np.std(pred_np)
            
            brightness_sim = 1 - abs(pred_brightness - self.criterion.reference_features['brightness'])
            contrast_sim = 1 - abs(pred_contrast - self.criterion.reference_features['contrast'])
            
            ref_similarity = (brightness_sim + contrast_sim) / 2
        
        # Combined quality score
        quality_score = (
            psnr_score / 40.0 * 0.25 +
            ssim_score * 0.25 +
            (1 - color_diff) * 0.2 +
            clarity_score * 0.15 +
            ref_similarity * 0.15
        )
        
        return quality_score
    
    def train(self, num_epochs=150):
        """Train the reference-guided model"""
        
        logger.info("Starting Reference-Guided Training")
        logger.info(f"Reference image: {self.reference_image_path}")
        logger.info(f"Device: {self.device}")
        logger.info("="*60)
        
        # Create datasets
        train_dataset = DehazingDataset('data/train/hazy', 'data/train/clear')
        val_dataset = DehazingDataset('data/val/hazy', 'data/val/clear')
        
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2)
        
        # Training loop
        for epoch in range(num_epochs):
            # Training phase
            train_loss = self.train_epoch(train_loader)
            
            # Validation phase
            if epoch % 5 == 0:
                val_loss, val_quality = self.validate_epoch(val_loader)
                
                # Save best model
                if val_quality > self.best_quality_score:
                    self.best_quality_score = val_quality
                    self.best_model_state = self.model.state_dict().copy()
                    
                    # Save checkpoint
                    os.makedirs('models/reference_guided_dehazing', exist_ok=True)
                    torch.save({
                        'model_state_dict': self.best_model_state,
                        'quality_score': self.best_quality_score,
                        'epoch': epoch,
                        'reference_image': self.reference_image_path
                    }, 'models/reference_guided_dehazing/reference_guided_model.pth')
                
                logger.info(f"Epoch {epoch+1}/{num_epochs}: "
                           f"Train Loss: {train_loss:.6f}, "
                           f"Val Loss: {val_loss:.6f}, "
                           f"Quality: {val_quality:.4f} "
                           f"(Best: {self.best_quality_score:.4f})")
                
                # Quality assessment
                if val_quality >= 0.85:
                    logger.info("✅ Excellent quality achieved!")
                elif val_quality >= 0.75:
                    logger.info("✅ Good quality achieved!")
                elif val_quality >= 0.65:
                    logger.info("⚠️  Acceptable quality")
                else:
                    logger.info("❌ Quality needs improvement")
            
            self.scheduler.step()
        
        logger.info("Training completed!")
        logger.info(f"Best Quality Score: {self.best_quality_score:.4f}")
        
        return self.model, self.best_quality_score

def train_reference_guided_model():
    """Train the reference-guided model using playground image"""
    
    # Reference images to try
    reference_images = [
        'test_images/playground_hazy.jpg',
        'test_hazy_image.jpg'
    ]
    
    # Find available reference image
    reference_image = None
    for img_path in reference_images:
        if os.path.exists(img_path):
            reference_image = img_path
            break
    
    if not reference_image:
        logger.error("No reference image found!")
        return None, 0.0
    
    # Create trainer
    trainer = ReferenceGuidedTrainer(reference_image)
    
    # Train model
    model, quality_score = trainer.train()
    
    return model, quality_score

if __name__ == "__main__":
    train_reference_guided_model()
