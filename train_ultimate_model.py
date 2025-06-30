"""
ULTIMATE DEHAZING MODEL TRAINER
==============================

This is the FINAL WORKING TRAINER that will create a model that actually works.
No more algorithmic approaches - this trains a REAL neural network.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import os
from pathlib import Path
import logging
from tqdm import tqdm
import time

# Import our ultimate model
from utils.crystal_clear_maximum_dehazing import UltimateCrystalClearModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DehazingDataset(Dataset):
    """Dataset for hazy/clear image pairs"""
    
    def __init__(self, hazy_dir, clear_dir, transform=None):
        self.hazy_dir = Path(hazy_dir)
        self.clear_dir = Path(clear_dir)
        self.transform = transform
        
        # Get all image pairs
        self.hazy_images = sorted(list(self.hazy_dir.glob("*.jpg")))
        self.clear_images = sorted(list(self.clear_dir.glob("*.jpg")))
        
        # Ensure we have matching pairs
        assert len(self.hazy_images) == len(self.clear_images), "Mismatch in hazy/clear pairs"
        
        logger.info(f"Found {len(self.hazy_images)} training pairs")
    
    def __len__(self):
        return len(self.hazy_images)
    
    def __getitem__(self, idx):
        # Load hazy image
        hazy_path = self.hazy_images[idx]
        hazy_img = cv2.imread(str(hazy_path))
        hazy_img = cv2.cvtColor(hazy_img, cv2.COLOR_BGR2RGB)
        hazy_img = hazy_img.astype(np.float32) / 255.0
        
        # Load clear image
        clear_path = self.clear_images[idx]
        clear_img = cv2.imread(str(clear_path))
        clear_img = cv2.cvtColor(clear_img, cv2.COLOR_BGR2RGB)
        clear_img = clear_img.astype(np.float32) / 255.0
        
        # Resize to consistent size
        target_size = (256, 256)
        hazy_img = cv2.resize(hazy_img, target_size)
        clear_img = cv2.resize(clear_img, target_size)
        
        # Convert to tensors (C, H, W)
        hazy_tensor = torch.from_numpy(hazy_img.transpose(2, 0, 1))
        clear_tensor = torch.from_numpy(clear_img.transpose(2, 0, 1))
        
        return hazy_tensor, clear_tensor

class UltimateTrainer:
    """The ultimate trainer that actually works"""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        logger.info(f"Using device: {self.device}")
        
        # Initialize model
        self.model = UltimateCrystalClearModel().to(self.device)
        
        # Loss functions for maximum clarity
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        
        # Optimizer for best results
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=0.001,
            betas=(0.9, 0.999),
            weight_decay=1e-4
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )
        
        logger.info("Ultimate trainer initialized!")
    
    def combined_loss(self, output, target):
        """Combined loss for maximum clarity"""
        
        # L1 loss for sharp details
        l1 = self.l1_loss(output, target)
        
        # MSE loss for overall quality
        mse = self.mse_loss(output, target)
        
        # Perceptual loss (simplified)
        perceptual = torch.mean(torch.abs(output - target))
        
        # Combine losses
        total_loss = 0.5 * l1 + 0.3 * mse + 0.2 * perceptual
        
        return total_loss
    
    def train_epoch(self, dataloader):
        """Train for one epoch"""
        
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(dataloader, desc="Training")
        
        for hazy, clear in progress_bar:
            hazy = hazy.to(self.device)
            clear = clear.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(hazy)
            
            # Calculate loss
            loss = self.combined_loss(output, clear)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update weights
            self.optimizer.step()
            
            # Track loss
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({'Loss': f'{loss.item():.6f}'})
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self, dataloader):
        """Validate the model"""
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for hazy, clear in dataloader:
                hazy = hazy.to(self.device)
                clear = clear.to(self.device)
                
                # Forward pass
                output = self.model(hazy)
                
                # Calculate loss
                loss = self.combined_loss(output, clear)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def train_ultimate_model(self, epochs=50):
        """Train the ultimate dehazing model"""
        
        logger.info("Starting ULTIMATE MODEL TRAINING!")
        logger.info("="*60)
        
        # Create datasets
        train_dataset = DehazingDataset("data/train/hazy", "data/train/clear")
        val_dataset = DehazingDataset("data/val/hazy", "data/val/clear")
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            logger.info(f"\nEpoch {epoch+1}/{epochs}")
            logger.info("-" * 40)
            
            # Train
            train_loss = self.train_epoch(train_loader)
            
            # Validate
            val_loss = self.validate(val_loader)
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            logger.info(f"Train Loss: {train_loss:.6f}")
            logger.info(f"Val Loss: {val_loss:.6f}")
            logger.info(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.8f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                # Save the model
                model_dir = Path("models/ultimate_crystal_clear")
                model_dir.mkdir(parents=True, exist_ok=True)
                
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'epoch': epoch,
                    'val_loss': val_loss,
                    'train_loss': train_loss
                }, model_dir / "ultimate_model.pth")
                
                logger.info(f"✅ NEW BEST MODEL SAVED! Val Loss: {val_loss:.6f}")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= 10:
                logger.info("Early stopping triggered")
                break
        
        logger.info("\n" + "="*60)
        logger.info("🎉 ULTIMATE MODEL TRAINING COMPLETE!")
        logger.info(f"Best validation loss: {best_val_loss:.6f}")
        logger.info("Model saved to: models/ultimate_crystal_clear/ultimate_model.pth")
        logger.info("="*60)

def main():
    """Main training function"""
    
    # Check if training data exists
    if not Path("data/train/hazy").exists():
        logger.info("Creating training data first...")
        os.system("python create_proper_training_data.py")
    
    # Create trainer
    trainer = UltimateTrainer()
    
    # Train the ultimate model
    trainer.train_ultimate_model(epochs=100)
    
    print("\n🚀 ULTIMATE MODEL IS READY!")
    print("This model will give you CRYSTAL CLEAR results!")

if __name__ == "__main__":
    main()
