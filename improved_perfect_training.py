"""
Improved Perfect Balanced Training
=================================

An improved training approach that achieves the perfect balance:
- Crystal clear visibility (strong clarity improvement)
- Natural appearance (not too aggressive)
- Professional quality matching your reference image
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImprovedPerfectNet(nn.Module):
    """Improved Perfect Balanced Dehazing Network"""
    
    def __init__(self):
        super(ImprovedPerfectNet, self).__init__()
        
        # Enhanced encoder for better feature extraction
        self.encoder1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.encoder2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.encoder3 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Enhanced attention mechanism
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(256, 64, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 256, 1),
            nn.Sigmoid()
        )
        
        # Decoder with skip connections
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.decoder2 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),  # 256 from skip connection
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.decoder1 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),  # 128 from skip connection
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Final layers for perfect balance
        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, 1),
            nn.Sigmoid()
        )
        
        # Refinement layer for natural appearance
        self.refinement = nn.Conv2d(6, 3, 1)  # 6 channels: 3 from dehazed + 3 from input
        
    def forward(self, x):
        # Store input and skip connections
        input_img = x
        
        # Encoder path
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        
        # Attention
        attention_weights = self.attention(enc3)
        attended = enc3 * attention_weights
        
        # Decoder path with skip connections
        up2 = self.upconv2(attended)
        dec2 = torch.cat([up2, enc2], dim=1)
        dec2 = self.decoder2(dec2)
        
        up1 = self.upconv1(dec2)
        dec1 = torch.cat([up1, enc1], dim=1)
        dec1 = self.decoder1(dec1)
        
        # Initial dehazing
        dehazed = self.final_conv(dec1)
        
        # Refinement with input for perfect balance
        combined = torch.cat([dehazed, input_img], dim=1)
        refined = self.refinement(combined)
        
        # Perfect balance: 80% dehazed + 20% original for crystal clarity with naturalness
        balanced = torch.sigmoid(refined) * 0.8 + input_img * 0.2
        
        return torch.clamp(balanced, 0, 1)

class ImprovedDataset(Dataset):
    """Improved dataset with data augmentation"""
    
    def __init__(self, hazy_dir, clear_dir):
        self.hazy_dir = hazy_dir
        self.clear_dir = clear_dir
        
        # Get image files
        hazy_files = [f for f in os.listdir(hazy_dir) if f.endswith(('.jpg', '.png'))]
        clear_files = [f for f in os.listdir(clear_dir) if f.endswith(('.jpg', '.png'))]
        
        # Match pairs
        self.pairs = []
        for hazy_file in hazy_files:
            if hazy_file in clear_files:
                self.pairs.append(hazy_file)
        
        # Augment dataset by repeating with variations
        self.augmented_pairs = []
        for pair in self.pairs:
            # Original
            self.augmented_pairs.append((pair, 'original'))
            # Brightness variations
            self.augmented_pairs.append((pair, 'bright'))
            self.augmented_pairs.append((pair, 'dark'))
            # Contrast variations
            self.augmented_pairs.append((pair, 'high_contrast'))
            self.augmented_pairs.append((pair, 'low_contrast'))
        
        logger.info(f"Found {len(self.pairs)} image pairs, augmented to {len(self.augmented_pairs)}")
    
    def __len__(self):
        return len(self.augmented_pairs)
    
    def __getitem__(self, idx):
        filename, augmentation = self.augmented_pairs[idx]
        
        # Load images
        hazy = cv2.imread(os.path.join(self.hazy_dir, filename))
        clear = cv2.imread(os.path.join(self.clear_dir, filename))
        
        # Convert BGR to RGB
        hazy = cv2.cvtColor(hazy, cv2.COLOR_BGR2RGB)
        clear = cv2.cvtColor(clear, cv2.COLOR_BGR2RGB)
        
        # Apply augmentation
        hazy, clear = self.apply_augmentation(hazy, clear, augmentation)
        
        # Resize
        hazy = cv2.resize(hazy, (256, 256))
        clear = cv2.resize(clear, (256, 256))
        
        # Normalize
        hazy = hazy.astype(np.float32) / 255.0
        clear = clear.astype(np.float32) / 255.0
        
        # To tensor
        hazy = torch.from_numpy(hazy.transpose(2, 0, 1))
        clear = torch.from_numpy(clear.transpose(2, 0, 1))
        
        return hazy, clear
    
    def apply_augmentation(self, hazy, clear, augmentation):
        """Apply data augmentation"""
        
        if augmentation == 'bright':
            factor = np.random.uniform(1.1, 1.3)
            hazy = np.clip(hazy * factor, 0, 255)
            clear = np.clip(clear * factor, 0, 255)
        elif augmentation == 'dark':
            factor = np.random.uniform(0.7, 0.9)
            hazy = np.clip(hazy * factor, 0, 255)
            clear = np.clip(clear * factor, 0, 255)
        elif augmentation == 'high_contrast':
            factor = np.random.uniform(1.1, 1.2)
            hazy = np.clip((hazy - 128) * factor + 128, 0, 255)
            clear = np.clip((clear - 128) * factor + 128, 0, 255)
        elif augmentation == 'low_contrast':
            factor = np.random.uniform(0.8, 0.9)
            hazy = np.clip((hazy - 128) * factor + 128, 0, 255)
            clear = np.clip((clear - 128) * factor + 128, 0, 255)
        
        return hazy.astype(np.uint8), clear.astype(np.uint8)

class ImprovedBalancedLoss(nn.Module):
    """Improved balanced loss for perfect results"""
    
    def __init__(self):
        super(ImprovedBalancedLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
    
    def forward(self, pred, target):
        # Basic reconstruction losses
        mse_loss = self.mse(pred, target)
        l1_loss = self.l1(pred, target)
        
        # Perceptual loss (simplified)
        pred_gray = 0.299 * pred[:, 0] + 0.587 * pred[:, 1] + 0.114 * pred[:, 2]
        target_gray = 0.299 * target[:, 0] + 0.587 * target[:, 1] + 0.114 * target[:, 2]
        perceptual_loss = self.l1(pred_gray, target_gray)
        
        # Color consistency loss
        pred_mean = torch.mean(pred, dim=(2, 3))
        target_mean = torch.mean(target, dim=(2, 3))
        color_loss = torch.mean(torch.abs(pred_mean - target_mean))
        
        # Edge preservation loss
        pred_edges = self.calculate_edges(pred)
        target_edges = self.calculate_edges(target)
        edge_loss = self.l1(pred_edges, target_edges)
        
        # Perfect balance combination
        total_loss = (
            0.3 * mse_loss +        # Basic reconstruction
            0.25 * l1_loss +        # Color preservation
            0.2 * perceptual_loss + # Visual quality
            0.15 * color_loss +     # Color consistency
            0.1 * edge_loss         # Edge preservation
        )
        
        return total_loss
    
    def calculate_edges(self, images):
        """Calculate edge maps for edge preservation"""
        # Convert to grayscale
        gray = 0.299 * images[:, 0:1] + 0.587 * images[:, 1:2] + 0.114 * images[:, 2:3]
        
        # Sobel edge detection
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(images.device)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(images.device)
        
        edges_x = torch.nn.functional.conv2d(gray, sobel_x, padding=1)
        edges_y = torch.nn.functional.conv2d(gray, sobel_y, padding=1)
        
        edges = torch.sqrt(edges_x**2 + edges_y**2)
        return edges

def train_improved_model():
    """Train the improved perfect balanced model"""
    
    logger.info("Starting Improved Perfect Balanced Training")
    logger.info("="*60)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Model
    model = ImprovedPerfectNet().to(device)
    logger.info("Improved Perfect Balanced Network created")
    
    # Dataset with augmentation
    dataset = ImprovedDataset('data/train/hazy', 'data/train/clear')
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Loss and optimizer
    criterion = ImprovedBalancedLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0008, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=80, eta_min=1e-6)
    
    # Training
    num_epochs = 80
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        
        for hazy, clear in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            hazy, clear = hazy.to(device), clear.to(device)
            
            optimizer.zero_grad()
            dehazed = model(hazy)
            loss = criterion(dehazed, clear)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item()
        
        scheduler.step()
        avg_loss = total_loss / len(dataloader)
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            os.makedirs('models/improved_perfect_balanced', exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'loss': best_loss,
                'epoch': epoch
            }, 'models/improved_perfect_balanced/improved_perfect_model.pth')
        
        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch+1}: Loss = {avg_loss:.6f} (Best: {best_loss:.6f})")
    
    logger.info("Training completed!")
    logger.info(f"Best Loss: {best_loss:.6f}")
    
    return model

def test_improved_model():
    """Test the improved model"""
    
    logger.info("Testing improved model...")
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ImprovedPerfectNet().to(device)
    
    model_path = 'models/improved_perfect_balanced/improved_perfect_model.pth'
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded model with loss: {checkpoint['loss']:.6f}")
    else:
        logger.warning("No trained model found, using untrained model")
    
    model.eval()
    
    # Test images
    test_images = [
        'test_hazy_image.jpg',
        'test_images/playground_hazy.jpg'
    ]
    
    for img_path in test_images:
        if not os.path.exists(img_path):
            continue
        
        # Load and preprocess
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_size = image.shape[:2]
        
        # Resize for model
        image_resized = cv2.resize(image, (256, 256))
        image_norm = image_resized.astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_norm.transpose(2, 0, 1)).unsqueeze(0).to(device)
        
        # Inference
        with torch.no_grad():
            dehazed_tensor = model(image_tensor)
        
        # Postprocess
        dehazed = dehazed_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        dehazed = np.clip(dehazed, 0, 1)
        dehazed = (dehazed * 255).astype(np.uint8)
        
        # Resize back
        dehazed = cv2.resize(dehazed, (original_size[1], original_size[0]))
        
        # Save result
        output_name = os.path.basename(img_path).replace('.jpg', '_improved_perfect_balanced.jpg')
        output_path = f'test_results/{output_name}'
        os.makedirs('test_results', exist_ok=True)
        
        dehazed_bgr = cv2.cvtColor(dehazed, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, dehazed_bgr)
        
        logger.info(f"Processed {img_path} -> {output_path}")
        
        # Create comparison
        comparison = np.hstack([image, dehazed])
        comparison_path = output_path.replace('.jpg', '_comparison.jpg')
        comparison_bgr = cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR)
        cv2.imwrite(comparison_path, comparison_bgr)
        
        logger.info(f"Comparison saved: {comparison_path}")

def main():
    """Main function"""
    
    # Train the improved model
    model = train_improved_model()
    
    # Test the model
    test_improved_model()
    
    logger.info("Improved Perfect Balanced Training Complete!")
    logger.info("Check test_results/ folder for improved output images")

if __name__ == "__main__":
    main()
