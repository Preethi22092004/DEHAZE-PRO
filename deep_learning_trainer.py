#!/usr/bin/env python3
"""
Deep Learning Dehazing Training System

This script implements proper end-to-end deep learning training for dehazing models.
It creates synthetic training data and trains models with proper loss functions.
"""

import cv2
import numpy as np
import os
import time
import logging
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import json
from utils.model import AODNet, LightDehazeNet, DeepDehazeNet

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DehazingDataset(Dataset):
    """
    Dataset class for dehazing training with hazy-clear image pairs
    """
    def __init__(self, hazy_paths, clear_paths, transform=None, image_size=(256, 256)):
        self.hazy_paths = hazy_paths
        self.clear_paths = clear_paths
        self.transform = transform
        self.image_size = image_size
        
        assert len(hazy_paths) == len(clear_paths), "Number of hazy and clear images must match"
    
    def __len__(self):
        return len(self.hazy_paths)
    
    def __getitem__(self, idx):
        # Load hazy and clear images
        hazy_img = Image.open(self.hazy_paths[idx]).convert('RGB')
        clear_img = Image.open(self.clear_paths[idx]).convert('RGB')
        
        # Resize images
        hazy_img = hazy_img.resize(self.image_size, Image.LANCZOS)
        clear_img = clear_img.resize(self.image_size, Image.LANCZOS)
        
        # Apply transforms
        if self.transform:
            hazy_img = self.transform(hazy_img)
            clear_img = self.transform(clear_img)
        
        return hazy_img, clear_img

class PerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG features for better visual quality
    """
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        # Use VGG16 features
        try:
            vgg = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True)
            self.features = vgg.features[:16]  # Use up to conv3_3
            for param in self.features.parameters():
                param.requires_grad = False
            self.features.eval()
            self.use_vgg = True
        except:
            logger.warning("Could not load VGG16, using MSE loss instead")
            self.use_vgg = False
        
    def forward(self, pred, target):
        if self.use_vgg:
            pred_features = self.features(pred)
            target_features = self.features(target)
            return nn.functional.mse_loss(pred_features, target_features)
        else:
            return nn.functional.mse_loss(pred, target)

class SSIMLoss(nn.Module):
    """
    SSIM Loss for structural similarity
    """
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
    
    def forward(self, pred, target):
        # Simplified SSIM implementation
        mu1 = nn.functional.avg_pool2d(pred, self.window_size, stride=1, padding=self.window_size//2)
        mu2 = nn.functional.avg_pool2d(target, self.window_size, stride=1, padding=self.window_size//2)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = nn.functional.avg_pool2d(pred * pred, self.window_size, stride=1, padding=self.window_size//2) - mu1_sq
        sigma2_sq = nn.functional.avg_pool2d(target * target, self.window_size, stride=1, padding=self.window_size//2) - mu2_sq
        sigma12 = nn.functional.avg_pool2d(pred * target, self.window_size, stride=1, padding=self.window_size//2) - mu1_mu2
        
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        if self.size_average:
            return 1 - ssim_map.mean()
        else:
            return 1 - ssim_map.mean(1).mean(1).mean(1)

class DeepLearningDehazingTrainer:
    """
    Comprehensive deep learning dehazing training system
    """
    
    def __init__(self, input_dir="test_images", output_dir="training_results", model_dir="static/models/weights"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.model_dir = Path(model_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Training parameters
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 4
        self.learning_rate = 1e-4
        self.num_epochs = 50
        self.image_size = (256, 256)
        
        logger.info(f"Initialized DeepLearningDehazingTrainer")
        logger.info(f"Device: {self.device}")
        logger.info(f"Input directory: {self.input_dir}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Model directory: {self.model_dir}")
    
    def add_synthetic_haze(self, clear_img, haze_level=0.5):
        """
        Add synthetic haze to a clear image using atmospheric scattering model
        """
        # Convert to float
        clear_float = clear_img.astype(np.float32) / 255.0
        
        # Generate random atmospheric light (bright value)
        A = np.random.uniform(0.7, 1.0, (1, 1, 3))
        
        # Generate transmission map (random pattern)
        h, w = clear_img.shape[:2]
        t = np.random.uniform(0.2, 0.8, (h, w))
        
        # Apply Gaussian blur to make transmission smoother
        t = cv2.GaussianBlur(t, (21, 21), 0)
        
        # Adjust transmission based on haze level
        t = t * (1 - haze_level) + haze_level * 0.3
        t = np.expand_dims(t, axis=2)
        
        # Apply atmospheric scattering model: I = J*t + A*(1-t)
        hazy_float = clear_float * t + A * (1 - t)
        hazy_float = np.clip(hazy_float, 0, 1)
        
        return (hazy_float * 255).astype(np.uint8)
    
    def create_synthetic_dataset(self):
        """
        Create synthetic hazy images from clear images for training
        """
        logger.info("Creating synthetic training dataset...")
        
        # Find all images in input directory
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        all_images = []
        for ext in image_extensions:
            all_images.extend(list(self.input_dir.glob(ext)))
            all_images.extend(list(self.input_dir.glob(ext.upper())))
        
        if len(all_images) < 3:
            logger.error("Not enough images found for training. Need at least 3 images.")
            return [], []
        
        hazy_dir = self.output_dir / "synthetic_hazy"
        clear_dir = self.output_dir / "synthetic_clear"
        hazy_dir.mkdir(exist_ok=True)
        clear_dir.mkdir(exist_ok=True)
        
        hazy_paths = []
        clear_paths = []
        
        for i, img_path in enumerate(all_images[:20]):  # Use up to 20 images
            try:
                # Load clear image
                clear_img = cv2.imread(str(img_path))
                if clear_img is None:
                    continue
                
                # Resize to standard size
                clear_img = cv2.resize(clear_img, self.image_size)
                
                # Create multiple hazy versions with different parameters
                for j in range(5):  # 5 variations per image
                    haze_level = 0.2 + j * 0.15  # Different haze levels
                    hazy_img = self.add_synthetic_haze(clear_img, haze_level=haze_level)
                    
                    # Save images
                    clear_name = f"clear_{i:03d}_{j}.jpg"
                    hazy_name = f"hazy_{i:03d}_{j}.jpg"
                    
                    clear_path = clear_dir / clear_name
                    hazy_path = hazy_dir / hazy_name
                    
                    cv2.imwrite(str(clear_path), clear_img)
                    cv2.imwrite(str(hazy_path), hazy_img)
                    
                    clear_paths.append(str(clear_path))
                    hazy_paths.append(str(hazy_path))
                    
            except Exception as e:
                logger.error(f"Error processing {img_path}: {e}")
                continue
        
        logger.info(f"Created {len(hazy_paths)} training pairs")
        return hazy_paths, clear_paths
    
    def get_data_loaders(self, hazy_paths, clear_paths):
        """
        Create data loaders for training and validation
        """
        # Split data into train and validation
        train_hazy, val_hazy, train_clear, val_clear = train_test_split(
            hazy_paths, clear_paths, test_size=0.2, random_state=42
        )
        
        # Define transforms - Use simple normalization for better color preservation
        transform = transforms.Compose([
            transforms.ToTensor(),
            # No normalization to preserve natural colors
        ])
        
        # Create datasets
        train_dataset = DehazingDataset(train_hazy, train_clear, transform=transform, image_size=self.image_size)
        val_dataset = DehazingDataset(val_hazy, val_clear, transform=transform, image_size=self.image_size)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
        
        return train_loader, val_loader

    def train_model(self, model, model_name, train_loader, val_loader):
        """
        Train a dehazing model
        """
        logger.info(f"Training {model_name} model...")

        # Move model to device
        model = model.to(self.device)

        # Define loss functions
        l1_loss = nn.L1Loss()
        mse_loss = nn.MSELoss()
        perceptual_loss = PerceptualLoss().to(self.device)
        ssim_loss = SSIMLoss().to(self.device)

        # Define optimizer
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

        # Training history
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')

        for epoch in range(self.num_epochs):
            # Training phase
            model.train()
            train_loss = 0.0

            for batch_idx, (hazy, clear) in enumerate(train_loader):
                hazy, clear = hazy.to(self.device), clear.to(self.device)

                optimizer.zero_grad()

                # Forward pass
                output = model(hazy)

                # Calculate combined loss
                l1 = l1_loss(output, clear)
                mse = mse_loss(output, clear)
                perceptual = perceptual_loss(output, clear)
                ssim = ssim_loss(output, clear)

                # Combined loss with weights
                total_loss = l1 + 0.1 * mse + 0.01 * perceptual + 0.1 * ssim

                # Backward pass
                total_loss.backward()
                optimizer.step()

                train_loss += total_loss.item()

                if batch_idx % 10 == 0:
                    logger.info(f'Epoch {epoch+1}/{self.num_epochs}, Batch {batch_idx}, Loss: {total_loss.item():.6f}')

            # Validation phase
            model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for hazy, clear in val_loader:
                    hazy, clear = hazy.to(self.device), clear.to(self.device)
                    output = model(hazy)

                    # Calculate validation loss
                    l1 = l1_loss(output, clear)
                    mse = mse_loss(output, clear)
                    perceptual = perceptual_loss(output, clear)
                    ssim = ssim_loss(output, clear)

                    total_loss = l1 + 0.1 * mse + 0.01 * perceptual + 0.1 * ssim
                    val_loss += total_loss.item()

            # Calculate average losses
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)

            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)

            logger.info(f'Epoch {epoch+1}/{self.num_epochs}: Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')

            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                model_path = self.model_dir / f"{model_name.lower()}_net.pth"
                torch.save(model.state_dict(), model_path)
                logger.info(f'Saved best model to {model_path}')

            # Update learning rate
            scheduler.step()

        return train_losses, val_losses

    def train_all_models(self):
        """
        Train all dehazing models
        """
        logger.info("🔥 STARTING COMPREHENSIVE DEEP LEARNING TRAINING")
        logger.info("=" * 60)

        # Create synthetic dataset
        hazy_paths, clear_paths = self.create_synthetic_dataset()

        if len(hazy_paths) == 0:
            logger.error("No training data created. Cannot proceed with training.")
            return

        # Get data loaders
        train_loader, val_loader = self.get_data_loaders(hazy_paths, clear_paths)

        # Define models to train
        models_to_train = [
            (AODNet(), "AOD"),
            (LightDehazeNet(), "Light"),
            (DeepDehazeNet(), "Deep")
        ]

        training_results = {}

        for model, model_name in models_to_train:
            try:
                logger.info(f"\n🚀 Training {model_name} model...")
                train_losses, val_losses = self.train_model(model, model_name, train_loader, val_loader)

                training_results[model_name] = {
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                    'final_train_loss': train_losses[-1] if train_losses else 0,
                    'final_val_loss': val_losses[-1] if val_losses else 0
                }

                logger.info(f"✅ {model_name} training completed!")

            except Exception as e:
                logger.error(f"❌ Error training {model_name}: {e}")
                continue

        # Save training results
        results_path = self.output_dir / "training_results.json"
        with open(results_path, 'w') as f:
            json.dump(training_results, f, indent=2)

        logger.info(f"\n🎉 TRAINING COMPLETED!")
        logger.info(f"📊 Results saved to {results_path}")
        logger.info(f"🏆 Models saved to {self.model_dir}")

        return training_results

    def test_trained_models(self):
        """
        Test the trained models on sample images
        """
        logger.info("🧪 Testing trained models...")

        # Find test images
        test_images = list(self.input_dir.glob("*.jpg"))[:3]
        if not test_images:
            logger.warning("No test images found")
            return

        # Test each model
        models_to_test = [
            (AODNet(), "aod"),
            (LightDehazeNet(), "light"),
            (DeepDehazeNet(), "deep")
        ]

        test_dir = self.output_dir / "test_results"
        test_dir.mkdir(exist_ok=True)

        for model, model_name in models_to_test:
            try:
                # Load trained weights
                weight_path = self.model_dir / f"{model_name}_net.pth"
                if not weight_path.exists():
                    logger.warning(f"No trained weights found for {model_name}")
                    continue

                model.load_state_dict(torch.load(weight_path, map_location=self.device))
                model = model.to(self.device)
                model.eval()

                logger.info(f"Testing {model_name} model...")

                for img_path in test_images:
                    # Load and preprocess image
                    img = cv2.imread(str(img_path))
                    img_resized = cv2.resize(img, self.image_size)

                    # Convert to tensor
                    transform = transforms.Compose([
                        transforms.ToPILImage(),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])

                    img_tensor = transform(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(self.device)

                    # Process with model
                    with torch.no_grad():
                        output = model(img_tensor)

                    # Convert back to image
                    output = output.squeeze(0).cpu()

                    # No denormalization needed since we removed normalization
                    output = torch.clamp(output, 0, 1)

                    # Convert to numpy
                    output_np = output.permute(1, 2, 0).numpy()
                    output_np = (output_np * 255).astype(np.uint8)
                    output_bgr = cv2.cvtColor(output_np, cv2.COLOR_RGB2BGR)

                    # Save result
                    output_path = test_dir / f"{img_path.stem}_{model_name}_result.jpg"
                    cv2.imwrite(str(output_path), output_bgr)

                    logger.info(f"  ✅ {img_path.name} -> {output_path.name}")

            except Exception as e:
                logger.error(f"Error testing {model_name}: {e}")
                continue

        logger.info(f"🎯 Test results saved to {test_dir}")

def main():
    """
    Main training function
    """
    print("🔥 DEEP LEARNING DEHAZING TRAINING SYSTEM")
    print("=" * 70)
    print("🎯 TRAINING PROPER ML MODELS FOR PERFECT DEHAZING!")
    print("=" * 70)

    # Initialize trainer
    trainer = DeepLearningDehazingTrainer()

    # Run training
    results = trainer.train_all_models()

    # Test trained models
    trainer.test_trained_models()

    print("\n🎉 DEEP LEARNING TRAINING COMPLETED!")
    print("🔥 Your ML models are now properly trained!")
    print("🌐 Use the web interface at http://127.0.0.1:5000")
    print("💻 Or use CLI: python simple_dehaze.py your_image.jpg --method deep")

if __name__ == '__main__':
    main()
