#!/usr/bin/env python3
"""
Advanced Dehazing Model Training System
Creates a well-trained model for dramatic fog/haze removal like shown in the reference image
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import os
import random
from skimage.metrics import structural_similarity as ssim
import logging
from typing import Tuple, List
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedDehazingNet(nn.Module):
    """
    Advanced CNN architecture for superior dehazing performance
    Based on successful architectures but optimized for dramatic haze removal
    """
    
    def __init__(self):
        super(AdvancedDehazingNet, self).__init__()
        
        # Encoder with attention mechanisms
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv6 = nn.Conv2d(256, 256, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.conv7 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv8 = nn.Conv2d(512, 512, 3, padding=1)
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Conv2d(512, 256, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, 1),
            nn.Sigmoid()
        )
        
        # Decoder with skip connections
        self.upconv1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv9 = nn.Conv2d(512, 256, 3, padding=1)  # 512 because of skip connection
        self.conv10 = nn.Conv2d(256, 256, 3, padding=1)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv11 = nn.Conv2d(256, 128, 3, padding=1)  # 256 because of skip connection
        self.conv12 = nn.Conv2d(128, 128, 3, padding=1)
        
        self.upconv3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv13 = nn.Conv2d(128, 64, 3, padding=1)   # 128 because of skip connection
        self.conv14 = nn.Conv2d(64, 64, 3, padding=1)
        
        # Final layers for transmission map and atmospheric light
        self.transmission_map = nn.Conv2d(64, 1, 3, padding=1)
        self.atmospheric_light = nn.Conv2d(64, 3, 3, padding=1)
        
        # Refinement network
        self.refine_conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.refine_conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.refine_conv3 = nn.Conv2d(32, 3, 3, padding=1)
        
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        # Store for skip connections
        x1 = self.relu(self.conv1(x))
        x1 = self.relu(self.conv2(x1))
        p1 = self.pool1(x1)
        
        x2 = self.relu(self.conv3(p1))
        x2 = self.relu(self.conv4(x2))
        p2 = self.pool2(x2)
        
        x3 = self.relu(self.conv5(p2))
        x3 = self.relu(self.conv6(x3))
        p3 = self.pool3(x3)
        
        # Bottleneck with attention
        x4 = self.relu(self.conv7(p3))
        x4 = self.relu(self.conv8(x4))
        
        # Apply attention
        attention_weights = self.attention(x4)
        x4 = x4 * attention_weights
        
        # Decoder with skip connections
        up1 = self.upconv1(x4)
        merge1 = torch.cat([up1, x3], dim=1)
        d1 = self.relu(self.conv9(merge1))
        d1 = self.relu(self.conv10(d1))
        
        up2 = self.upconv2(d1)
        merge2 = torch.cat([up2, x2], dim=1)
        d2 = self.relu(self.conv11(merge2))
        d2 = self.relu(self.conv12(d2))
        
        up3 = self.upconv3(d2)
        merge3 = torch.cat([up3, x1], dim=1)
        d3 = self.relu(self.conv13(merge3))
        d3 = self.relu(self.conv14(d3))
        
        # Generate transmission map and atmospheric light
        t = self.sigmoid(self.transmission_map(d3))
        t = torch.clamp(t, 0.1, 1.0)  # Ensure minimum transmission
        
        A = self.sigmoid(self.atmospheric_light(d3))
        
        # Apply atmospheric scattering model inverse
        # J = (I - A) / t + A, so we need to solve for J
        dehazed = (x - A) / t + A
        
        # Refinement pass
        refined = self.relu(self.refine_conv1(dehazed))
        refined = self.relu(self.refine_conv2(refined))
        refined = self.tanh(self.refine_conv3(refined))
        
        # Final output with residual connection
        output = dehazed + 0.1 * refined
        output = torch.clamp(output, 0, 1)
        
        return output, t, A

class SyntheticHazeDataset(Dataset):
    """
    Dataset that creates synthetic hazy images from clear images
    """
    
    def __init__(self, clear_images_dir: str, num_samples: int = 1000, image_size: Tuple[int, int] = (256, 256)):
        self.clear_images_dir = clear_images_dir
        self.num_samples = num_samples
        self.image_size = image_size
        self.clear_images = []
        
        # If no clear images directory, generate synthetic data
        if not os.path.exists(clear_images_dir) or len(os.listdir(clear_images_dir)) == 0:
            logger.info("No clear images found, generating synthetic clear images...")
            self.generate_synthetic_clear_images()
        else:
            # Load existing clear images
            for img_file in os.listdir(clear_images_dir):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(clear_images_dir, img_file)
                    self.clear_images.append(img_path)
        
    def generate_synthetic_clear_images(self):
        """Generate synthetic clear images for training"""
        os.makedirs(self.clear_images_dir, exist_ok=True)
        
        for i in range(self.num_samples // 4):  # Generate base images
            # Create diverse synthetic scenes
            img = self.create_synthetic_scene(i)
            img_path = os.path.join(self.clear_images_dir, f"synthetic_clear_{i:04d}.jpg")
            cv2.imwrite(img_path, img)
            self.clear_images.append(img_path)
    
    def create_synthetic_scene(self, seed: int) -> np.ndarray:
        """Create a synthetic clear scene"""
        np.random.seed(seed)
        random.seed(seed)
        
        img = np.zeros((*self.image_size, 3), dtype=np.uint8)
        
        # Sky gradient
        sky_height = random.randint(50, 120)
        for y in range(sky_height):
            intensity = int(255 * (1 - y / sky_height * 0.3))
            color_variation = random.randint(-20, 20)
            sky_color = [
                max(0, min(255, intensity + color_variation)),
                max(0, min(255, intensity - 10 + color_variation)),
                max(0, min(255, intensity - 30 + color_variation))
            ]
            img[y, :] = sky_color
        
        # Ground
        ground_color = [
            random.randint(50, 150),
            random.randint(80, 180),
            random.randint(40, 120)
        ]
        img[sky_height:, :] = ground_color
        
        # Add buildings/objects
        num_objects = random.randint(3, 8)
        for _ in range(num_objects):
            x1 = random.randint(0, self.image_size[1] - 50)
            y1 = random.randint(sky_height - 20, self.image_size[0] - 30)
            x2 = random.randint(x1 + 20, min(x1 + 80, self.image_size[1]))
            y2 = random.randint(y1 + 30, self.image_size[0])
            
            object_color = [random.randint(30, 200) for _ in range(3)]
            cv2.rectangle(img, (x1, y1), (x2, y2), object_color, -1)
            
            # Add windows
            if random.random() > 0.5:
                for wy in range(y1 + 10, y2 - 10, 20):
                    for wx in range(x1 + 10, x2 - 10, 15):
                        if wx + 8 < x2 and wy + 12 < y2:
                            cv2.rectangle(img, (wx, wy), (wx + 8, wy + 12), 
                                        [random.randint(150, 255), random.randint(150, 255), random.randint(100, 200)], -1)
        
        # Add texture and noise
        noise = np.random.normal(0, 5, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        return img
    
    def apply_synthetic_haze(self, clear_img: np.ndarray) -> np.ndarray:
        """Apply realistic haze to clear image"""
        h, w = clear_img.shape[:2]
        
        # Create depth-based transmission map
        center_x, center_y = w // 2, h // 2
        y, x = np.ogrid[:h, :w]
        
        # Create more complex depth patterns
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_distance = np.sqrt(center_x**2 + center_y**2)
        
        # Add some randomness to depth
        depth_noise = np.random.normal(0, 0.1, (h, w))
        normalized_distance = distance / max_distance + depth_noise
        normalized_distance = np.clip(normalized_distance, 0, 1)
        
        # Vary haze intensity
        beta = random.uniform(0.8, 3.0)  # Scattering coefficient
        transmission = np.exp(-beta * normalized_distance)
        transmission = np.maximum(transmission, random.uniform(0.05, 0.2))
        
        # Atmospheric light with some variation
        base_atmospheric = random.uniform(0.7, 0.95)
        atmospheric_light = np.array([
            base_atmospheric + random.uniform(-0.1, 0.1),
            base_atmospheric + random.uniform(-0.1, 0.1),
            base_atmospheric + random.uniform(-0.15, 0.05)
        ]) * 255
        
        # Apply atmospheric scattering model
        clear_float = clear_img.astype(np.float32)
        transmission_3d = transmission[:, :, np.newaxis]
        
        hazy_img = (clear_float * transmission_3d + 
                   atmospheric_light * (1 - transmission_3d))
        
        hazy_img = np.clip(hazy_img, 0, 255).astype(np.uint8)
        
        return hazy_img
    
    def __len__(self):
        return max(len(self.clear_images) * 4, self.num_samples)  # Augment data
    
    def __getitem__(self, idx):
        # Get base clear image
        clear_idx = idx % len(self.clear_images)
        clear_img = cv2.imread(self.clear_images[clear_idx])
        
        # Resize to target size
        clear_img = cv2.resize(clear_img, self.image_size)
        
        # Apply random augmentations to clear image
        if random.random() > 0.5:
            clear_img = cv2.flip(clear_img, 1)  # Horizontal flip
        
        # Random brightness/contrast adjustment
        if random.random() > 0.5:
            alpha = random.uniform(0.8, 1.2)  # Contrast
            beta = random.randint(-20, 20)    # Brightness
            clear_img = cv2.convertScaleAbs(clear_img, alpha=alpha, beta=beta)
        
        # Generate hazy version
        hazy_img = self.apply_synthetic_haze(clear_img)
        
        # Convert to tensors and normalize
        clear_tensor = torch.from_numpy(clear_img.transpose(2, 0, 1)).float() / 255.0
        hazy_tensor = torch.from_numpy(hazy_img.transpose(2, 0, 1)).float() / 255.0
        
        return hazy_tensor, clear_tensor

class DehazingLoss(nn.Module):
    """
    Custom loss function combining multiple metrics for better dehazing
    """
    
    def __init__(self):
        super(DehazingLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
        
    def forward(self, pred, target, transmission=None, atmospheric_light=None):
        # Primary reconstruction loss
        mse_loss = self.mse(pred, target)
        l1_loss = self.l1(pred, target)
        
        # Perceptual loss (simple edge-based)
        pred_gray = torch.mean(pred, dim=1, keepdim=True)
        target_gray = torch.mean(target, dim=1, keepdim=True)
        
        pred_edges = self.compute_edge_loss(pred_gray)
        target_edges = self.compute_edge_loss(target_gray)
        edge_loss = self.mse(pred_edges, target_edges)
        
        # Color consistency loss
        color_loss = self.compute_color_loss(pred, target)
        
        # Total loss
        total_loss = (mse_loss + 
                     0.5 * l1_loss + 
                     0.3 * edge_loss + 
                     0.2 * color_loss)
        
        return total_loss
    
    def compute_edge_loss(self, img):
        """Compute edge map using Sobel operators"""
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        
        if img.is_cuda:
            sobel_x = sobel_x.cuda()
            sobel_y = sobel_y.cuda()
        
        edge_x = F.conv2d(img, sobel_x, padding=1)
        edge_y = F.conv2d(img, sobel_y, padding=1)
        
        edges = torch.sqrt(edge_x**2 + edge_y**2)
        return edges
    
    def compute_color_loss(self, pred, target):
        """Compute color consistency loss"""
        pred_mean = torch.mean(pred, dim=[2, 3])
        target_mean = torch.mean(target, dim=[2, 3])
        return self.mse(pred_mean, target_mean)

def train_advanced_dehazing_model(
    model_save_path: str = "advanced_dehazing_model.pth",
    num_epochs: int = 100,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    device: str = "cpu"
):
    """
    Train the advanced dehazing model
    """
    logger.info("🚀 Starting Advanced Dehazing Model Training...")
    
    # Setup device
    device = torch.device(device)
    logger.info(f"Using device: {device}")
    
    # Create dataset
    dataset = SyntheticHazeDataset(
        clear_images_dir="training_data/clear_images",
        num_samples=2000,
        image_size=(256, 256)
    )
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    # Create model
    model = AdvancedDehazingNet().to(device)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Setup training
    criterion = DehazingLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    # Training loop
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, (hazy_imgs, clear_imgs) in enumerate(dataloader):
            hazy_imgs = hazy_imgs.to(device)
            clear_imgs = clear_imgs.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            dehazed_imgs, transmission, atmospheric_light = model(hazy_imgs)
            
            # Calculate loss
            loss = criterion(dehazed_imgs, clear_imgs, transmission, atmospheric_light)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 20 == 0:
                logger.info(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        avg_loss = epoch_loss / num_batches
        scheduler.step(avg_loss)
        
        logger.info(f"Epoch {epoch+1}/{num_epochs} - Average Loss: {avg_loss:.4f}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, model_save_path)
            logger.info(f"✅ New best model saved with loss: {best_loss:.4f}")
        
        # Save checkpoint every 25 epochs
        if (epoch + 1) % 25 == 0:
            checkpoint_path = f"checkpoint_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    logger.info(f"🎉 Training completed! Best model saved as: {model_save_path}")
    return model_save_path

if __name__ == "__main__":
    # Train the model
    model_path = train_advanced_dehazing_model(
        model_save_path="weights/advanced_dehazing_model.pth",
        num_epochs=150,
        batch_size=4,  # Adjust based on your GPU memory
        learning_rate=1e-4,
        device="cpu"  # Change to "cuda" if you have GPU
    )
    
    print(f"✅ Training completed! Model saved to: {model_path}")
    print("🔧 You can now use this model for superior dehazing results!")
