#!/usr/bin/env python3
"""
Reference Match Model Training Script
===================================

Train the ultra-advanced dehazing model to match reference image quality.
This script implements state-of-the-art training techniques for maximum clarity.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import cv2
import numpy as np
import os
import logging
from typing import Tuple, Dict, List
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils.reference_match_dehazing import ReferenceMatchDehazingNet
import torchvision.transforms as transforms
from PIL import Image
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SyntheticHazeDataset(Dataset):
    """Generate synthetic hazy images for training"""
    
    def __init__(self, clear_images_dir: str, transform=None, num_samples=1000):
        self.clear_images_dir = clear_images_dir
        self.transform = transform
        self.num_samples = num_samples
        
        # Get all clear images
        self.clear_images = []
        if os.path.exists(clear_images_dir):
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                import glob
                self.clear_images.extend(glob.glob(os.path.join(clear_images_dir, ext)))
        
        # If no clear images, create some synthetic ones
        if not self.clear_images:
            logger.warning("No clear images found, will generate synthetic data")
            self.clear_images = ['synthetic'] * num_samples
    
    def __len__(self):
        return self.num_samples
    
    def add_synthetic_haze(self, image: np.ndarray) -> np.ndarray:
        """Add realistic synthetic haze to clear image"""
        height, width = image.shape[:2]
        
        # Random haze parameters
        beta = random.uniform(0.5, 2.0)  # Scattering coefficient
        A = random.uniform(0.7, 1.0)     # Atmospheric light
        
        # Create depth map (random gradient)
        depth = np.random.rand(height, width)
        depth = cv2.GaussianBlur(depth, (21, 21), 0)
        depth = (depth - depth.min()) / (depth.max() - depth.min())
        
        # Apply haze model: I(x) = J(x) * t(x) + A * (1 - t(x))
        transmission = np.exp(-beta * depth)
        transmission = np.stack([transmission] * 3, axis=2)
        
        # Normalize image
        image_norm = image.astype(np.float32) / 255.0
        
        # Apply haze
        hazy = image_norm * transmission + A * (1 - transmission)
        hazy = np.clip(hazy * 255, 0, 255).astype(np.uint8)
        
        return hazy
    
    def __getitem__(self, idx):
        if self.clear_images[0] == 'synthetic':
            # Generate synthetic clear image
            height, width = 256, 256
            clear = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            clear = cv2.GaussianBlur(clear, (5, 5), 0)
        else:
            # Load real clear image
            img_path = self.clear_images[idx % len(self.clear_images)]
            clear = cv2.imread(img_path)
            if clear is None:
                clear = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            else:
                clear = cv2.resize(clear, (256, 256))
        
        # Generate hazy version
        hazy = self.add_synthetic_haze(clear)
        
        # Convert to tensors
        if self.transform:
            clear = self.transform(clear)
            hazy = self.transform(hazy)
        else:
            clear = torch.from_numpy(clear.astype(np.float32) / 255.0).permute(2, 0, 1)
            hazy = torch.from_numpy(hazy.astype(np.float32) / 255.0).permute(2, 0, 1)
        
        return hazy, clear

class PerceptualLoss(nn.Module):
    """Perceptual loss for better visual quality"""
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        # Use a simple approximation of perceptual loss
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
    
    def forward(self, pred, target):
        # Combine MSE and L1 loss
        mse_loss = self.mse(pred, target)
        l1_loss = self.l1(pred, target)
        
        # Edge-aware loss
        pred_edges = self.sobel_edges(pred)
        target_edges = self.sobel_edges(target)
        edge_loss = self.mse(pred_edges, target_edges)
        
        return mse_loss + 0.1 * l1_loss + 0.2 * edge_loss
    
    def sobel_edges(self, x):
        """Compute Sobel edges"""
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        
        if x.is_cuda:
            sobel_x = sobel_x.cuda()
            sobel_y = sobel_y.cuda()
        
        # Convert to grayscale
        gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        
        edges_x = torch.nn.functional.conv2d(gray, sobel_x, padding=1)
        edges_y = torch.nn.functional.conv2d(gray, sobel_y, padding=1)
        
        edges = torch.sqrt(edges_x**2 + edges_y**2)
        return edges

def train_reference_match_model():
    """Train the reference match dehazing model"""
    
    # Configuration
    config = {
        'batch_size': 4,
        'learning_rate': 0.0001,
        'num_epochs': 100,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'save_dir': 'static/models/weights',
        'log_interval': 10,
        'save_interval': 20,
        'num_samples': 2000
    }
    
    logger.info(f"Training configuration: {config}")
    
    # Create directories
    os.makedirs(config['save_dir'], exist_ok=True)
    
    # Create dataset
    dataset = SyntheticHazeDataset(
        clear_images_dir='data/clear',  # Will use synthetic if not found
        num_samples=config['num_samples']
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        num_workers=2
    )
    
    # Initialize model
    model = ReferenceMatchDehazingNet()
    model = model.to(config['device'])
    
    # Loss function and optimizer
    criterion = PerceptualLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    
    # Training loop
    model.train()
    best_loss = float('inf')
    
    logger.info("Starting training...")
    
    for epoch in range(config['num_epochs']):
        epoch_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{config["num_epochs"]}')
        
        for batch_idx, (hazy, clear) in enumerate(progress_bar):
            hazy = hazy.to(config['device'])
            clear = clear.to(config['device'])
            
            # Forward pass
            optimizer.zero_grad()
            output = model(hazy)
            loss = criterion(output, clear)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})
            
            # Log progress
            if batch_idx % config['log_interval'] == 0:
                logger.info(f'Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        # Update learning rate
        scheduler.step()
        
        # Calculate average loss
        avg_loss = epoch_loss / len(dataloader)
        logger.info(f'Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}')
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            model_path = os.path.join(config['save_dir'], 'reference_match_net.pth')
            torch.save(model.state_dict(), model_path)
            logger.info(f'New best model saved with loss: {best_loss:.4f}')
        
        # Save checkpoint
        if (epoch + 1) % config['save_interval'] == 0:
            checkpoint_path = os.path.join(config['save_dir'], f'reference_match_net_epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f'Checkpoint saved: {checkpoint_path}')
    
    logger.info("Training completed!")
    
    # Save final model
    final_model_path = os.path.join(config['save_dir'], 'reference_match_net_final.pth')
    torch.save(model.state_dict(), final_model_path)
    logger.info(f'Final model saved: {final_model_path}')

if __name__ == '__main__':
    train_reference_match_model()
