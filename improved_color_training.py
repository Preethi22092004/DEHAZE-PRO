#!/usr/bin/env python3
"""
Improved Color-Preserving Dehazing Training
Fixes the purple tint issue and improves training for crystal clear results
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from pathlib import Path
import logging
from PIL import Image
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ColorPreservingLoss(nn.Module):
    """Custom loss function that preserves natural colors"""
    
    def __init__(self):
        super(ColorPreservingLoss, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        
    def forward(self, output, target):
        # Basic reconstruction loss
        l1 = self.l1_loss(output, target)
        mse = self.mse_loss(output, target)
        
        # Color consistency loss - ensure color channels are balanced
        output_mean = torch.mean(output, dim=[2, 3])  # [B, C]
        target_mean = torch.mean(target, dim=[2, 3])  # [B, C]
        color_loss = self.l1_loss(output_mean, target_mean)
        
        # Prevent color shift by ensuring RGB ratios are preserved
        output_ratios = output_mean / (torch.sum(output_mean, dim=1, keepdim=True) + 1e-8)
        target_ratios = target_mean / (torch.sum(target_mean, dim=1, keepdim=True) + 1e-8)
        ratio_loss = self.l1_loss(output_ratios, target_ratios)
        
        # Edge preservation loss
        def sobel_edges(x):
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(x.device)
            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(x.device)
            
            edges_x = F.conv2d(x.mean(dim=1, keepdim=True), sobel_x, padding=1)
            edges_y = F.conv2d(x.mean(dim=1, keepdim=True), sobel_y, padding=1)
            return torch.sqrt(edges_x**2 + edges_y**2 + 1e-8)
        
        output_edges = sobel_edges(output)
        target_edges = sobel_edges(target)
        edge_loss = self.l1_loss(output_edges, target_edges)
        
        # Combine losses
        total_loss = l1 + 0.1 * mse + 0.2 * color_loss + 0.3 * ratio_loss + 0.1 * edge_loss
        
        return total_loss

class ImprovedDehazingDataset(Dataset):
    """Dataset with better color preservation"""
    
    def __init__(self, hazy_dir, clear_dir, image_size=(256, 256)):
        self.hazy_dir = Path(hazy_dir)
        self.clear_dir = Path(clear_dir)
        self.image_size = image_size
        
        # Get image pairs
        self.hazy_images = sorted(list(self.hazy_dir.glob("*.jpg")))
        self.clear_images = sorted(list(self.clear_dir.glob("*.jpg")))
        
        # Simple transforms without normalization
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        logger.info(f"Found {len(self.hazy_images)} image pairs")
    
    def __len__(self):
        return len(self.hazy_images)
    
    def __getitem__(self, idx):
        # Load images
        hazy_path = self.hazy_images[idx]
        clear_path = self.clear_images[idx]
        
        hazy_img = Image.open(hazy_path).convert('RGB')
        clear_img = Image.open(clear_path).convert('RGB')
        
        # Resize
        hazy_img = hazy_img.resize(self.image_size, Image.LANCZOS)
        clear_img = clear_img.resize(self.image_size, Image.LANCZOS)
        
        # Convert to tensors
        hazy_tensor = self.transform(hazy_img)
        clear_tensor = self.transform(clear_img)
        
        return hazy_tensor, clear_tensor

class ImprovedColorTrainer:
    """Improved trainer for color-preserving dehazing"""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        logger.info(f"Using device: {self.device}")
    
    def train_model(self, model, train_loader, val_loader, num_epochs=50, lr=0.001):
        """Train the model with improved color preservation"""
        
        model = model.to(self.device)
        
        # Use custom color-preserving loss
        criterion = ColorPreservingLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
        
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            
            for batch_idx, (hazy, clear) in enumerate(train_loader):
                hazy = hazy.to(self.device)
                clear = clear.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                output = model(hazy)
                loss = criterion(output, clear)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                train_loss += loss.item()
                
                if batch_idx % 10 == 0:
                    logger.info(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, Loss: {loss.item():.6f}')
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for hazy, clear in val_loader:
                    hazy = hazy.to(self.device)
                    clear = clear.to(self.device)
                    
                    output = model(hazy)
                    loss = criterion(output, clear)
                    val_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            logger.info(f'Epoch {epoch+1}/{num_epochs}: Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')
            
            # Learning rate scheduling
            scheduler.step(avg_val_loss)
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), 'models/improved_color_model.pth')
                logger.info(f'New best model saved with validation loss: {best_val_loss:.6f}')
        
        return model

def create_improved_training_data():
    """Create better training data with natural color preservation"""
    
    logger.info("Creating improved training data...")
    
    # Create directories
    train_dir = Path("training_data/improved")
    train_dir.mkdir(parents=True, exist_ok=True)
    
    hazy_dir = train_dir / "hazy"
    clear_dir = train_dir / "clear"
    hazy_dir.mkdir(exist_ok=True)
    clear_dir.mkdir(exist_ok=True)
    
    # Generate synthetic training data with better color preservation
    sample_images = list(Path("test_images").glob("*.jpg"))[:10]
    
    for i, img_path in enumerate(sample_images):
        try:
            # Load image
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            # Resize
            img = cv2.resize(img, (256, 256))
            
            # Create multiple hazy versions with different parameters
            for j in range(3):  # 3 variations per image
                # Create realistic haze with better color preservation
                haze_density = 0.3 + j * 0.2
                
                # Convert to float
                img_float = img.astype(np.float32) / 255.0
                
                # Add atmospheric light (more realistic)
                atmospheric_light = np.array([0.8, 0.85, 0.9])  # Slightly blue-tinted
                
                # Create transmission map
                transmission = np.exp(-haze_density * np.ones_like(img_float[:,:,0]))
                transmission = np.stack([transmission] * 3, axis=2)
                
                # Apply haze model: I = J * t + A * (1 - t)
                hazy_img = img_float * transmission + atmospheric_light * (1 - transmission)
                hazy_img = np.clip(hazy_img, 0, 1)
                
                # Convert back to uint8
                clear_img_uint8 = (img_float * 255).astype(np.uint8)
                hazy_img_uint8 = (hazy_img * 255).astype(np.uint8)
                
                # Save images
                clear_name = f"clear_{i:03d}_{j}.jpg"
                hazy_name = f"hazy_{i:03d}_{j}.jpg"
                
                cv2.imwrite(str(clear_dir / clear_name), clear_img_uint8)
                cv2.imwrite(str(hazy_dir / hazy_name), hazy_img_uint8)
        
        except Exception as e:
            logger.error(f"Error processing {img_path}: {e}")
            continue
    
    logger.info(f"Created improved training data in {train_dir}")
    return hazy_dir, clear_dir

if __name__ == "__main__":
    # Create improved training data
    hazy_dir, clear_dir = create_improved_training_data()
    
    # Create dataset and data loaders
    dataset = ImprovedDehazingDataset(hazy_dir, clear_dir)
    
    # Split into train and validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    # Load existing model
    from utils.model import LightDehazeNet
    model = LightDehazeNet()
    
    # Train with improved color preservation
    trainer = ImprovedColorTrainer()
    trained_model = trainer.train_model(model, train_loader, val_loader, num_epochs=30)
    
    logger.info("✅ Improved color training completed!")
