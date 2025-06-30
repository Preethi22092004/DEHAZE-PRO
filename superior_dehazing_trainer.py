#!/usr/bin/env python3
"""
Advanced Deep Learning Dehazing Model Training System
Creates a state-of-the-art dehazing model that effectively clears fog and haze
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import os
import logging
from tqdm import tqdm
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AttentionBlock(nn.Module):
    """Channel and spatial attention mechanism"""
    def __init__(self, channels):
        super(AttentionBlock, self).__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 8, channels, 1),
            nn.Sigmoid()
        )
        
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(channels, 1, 7, padding=3),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Channel attention
        ca = self.channel_attention(x)
        x = x * ca
        
        # Spatial attention
        sa = self.spatial_attention(x)
        x = x * sa
        
        return x

class ResidualDenseBlock(nn.Module):
    """Residual Dense Block for better feature learning"""
    def __init__(self, channels, growth_rate=32):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, growth_rate, 3, padding=1)
        self.conv2 = nn.Conv2d(channels + growth_rate, growth_rate, 3, padding=1)
        self.conv3 = nn.Conv2d(channels + 2 * growth_rate, growth_rate, 3, padding=1)
        self.conv4 = nn.Conv2d(channels + 3 * growth_rate, growth_rate, 3, padding=1)
        self.conv5 = nn.Conv2d(channels + 4 * growth_rate, channels, 3, padding=1)
        
    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(torch.cat([x, x1], 1)))
        x3 = F.relu(self.conv3(torch.cat([x, x1, x2], 1)))
        x4 = F.relu(self.conv4(torch.cat([x, x1, x2, x3], 1)))
        x5 = self.conv5(torch.cat([x, x1, x2, x3, x4], 1))
        return x5 + x

class SuperiorDehazingNet(nn.Module):
    """Superior dehazing network with advanced architecture"""
    
    def __init__(self):
        super(SuperiorDehazingNet, self).__init__()
        
        # Initial feature extraction
        self.initial_conv = nn.Conv2d(3, 64, 3, padding=1)
        
        # Encoder with progressive downsampling
        self.encoder1 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1, stride=2),
            nn.ReLU(inplace=True),
            AttentionBlock(128)
        )
        
        self.encoder2 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1, stride=2),
            nn.ReLU(inplace=True),
            AttentionBlock(256)
        )
        
        # Residual Dense Blocks for feature processing
        self.rdb_blocks = nn.ModuleList([
            ResidualDenseBlock(256) for _ in range(8)
        ])
        
        # Multi-scale feature fusion
        self.fusion_conv = nn.Conv2d(256, 256, 1)
        
        # Decoder with progressive upsampling
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            AttentionBlock(128)
        )
        
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            AttentionBlock(64)
        )
        
        # Final output layer
        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Tanh()
        )
        
        # Global residual connection
        self.global_residual_weight = nn.Parameter(torch.tensor(0.1))
        
    def forward(self, x):
        # Store input for global residual
        input_x = x
        
        # Initial feature extraction
        x = F.relu(self.initial_conv(x))
        
        # Encoder
        x1 = self.encoder1(x)
        x2 = self.encoder2(x1)
        
        # Residual Dense Blocks processing
        rdb_out = x2
        for rdb in self.rdb_blocks:
            rdb_out = rdb(rdb_out)
        
        # Feature fusion
        fused = self.fusion_conv(rdb_out)
        
        # Decoder
        up1 = self.decoder1(fused)
        up2 = self.decoder2(up1)
        
        # Final output
        output = self.final_conv(up2)
        
        # Normalize to [0,1]
        output = (output + 1) / 2
        
        # Global residual connection with learnable weight
        output = output + self.global_residual_weight * input_x
        output = torch.clamp(output, 0, 1)
        
        return output

class AdvancedSyntheticDataset(Dataset):
    """Advanced synthetic dataset for training"""
    
    def __init__(self, num_samples=2000, size=(256, 256)):
        self.num_samples = num_samples
        self.size = size
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Generate synthetic clear image
        clear_img = self.generate_complex_scene()
        
        # Apply realistic atmospheric scattering
        hazy_img = self.apply_realistic_haze(clear_img)
        
        # Convert to tensors
        clear_tensor = torch.FloatTensor(clear_img).permute(2, 0, 1)
        hazy_tensor = torch.FloatTensor(hazy_img).permute(2, 0, 1)
        
        return hazy_tensor, clear_tensor
    
    def generate_complex_scene(self):
        """Generate complex synthetic scene"""
        img = np.zeros((*self.size, 3), dtype=np.float32)
        
        # Sky gradient
        for y in range(self.size[0] // 3):
            intensity = 0.8 - 0.3 * (y / (self.size[0] // 3))
            img[y, :] = [intensity * 0.7, intensity * 0.8, intensity]
        
        # Ground
        ground_color = np.random.uniform(0.1, 0.4, 3)
        img[self.size[0] // 3:, :] = ground_color
        
        # Add multiple objects with different distances
        num_objects = np.random.randint(8, 15)
        for _ in range(num_objects):
            self.add_random_object(img)
        
        # Add texture and details
        noise = np.random.normal(0, 0.02, img.shape)
        img = np.clip(img + noise, 0, 1)
        
        return img
    
    def add_random_object(self, img):
        """Add random object to scene"""
        obj_type = np.random.choice(['rectangle', 'circle', 'polygon'])
        color = np.random.uniform(0.2, 0.9, 3)
        
        if obj_type == 'rectangle':
            x1 = np.random.randint(0, self.size[1] - 50)
            y1 = np.random.randint(self.size[0] // 4, self.size[0] - 50)
            w = np.random.randint(20, 80)
            h = np.random.randint(20, 80)
            
            x2 = min(x1 + w, self.size[1])
            y2 = min(y1 + h, self.size[0])
            
            img[y1:y2, x1:x2] = color
            
        elif obj_type == 'circle':
            center_x = np.random.randint(30, self.size[1] - 30)
            center_y = np.random.randint(self.size[0] // 4, self.size[0] - 30)
            radius = np.random.randint(10, 40)
            
            y, x = np.ogrid[:self.size[0], :self.size[1]]
            mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
            img[mask] = color
    
    def apply_realistic_haze(self, clear_img):
        """Apply realistic atmospheric scattering"""
        h, w = clear_img.shape[:2]
        
        # Create depth-based transmission map
        depth_map = np.zeros((h, w))
        
        # Sky has maximum depth (farthest)
        depth_map[:h//3, :] = np.random.uniform(0.8, 1.0)
        
        # Objects at various depths
        depth_map[h//3:, :] = np.random.uniform(0.2, 0.7, (h - h//3, w))
        
        # Add distance-based depth variation
        center_x = w // 2
        for y in range(h):
            for x in range(w):
                distance_factor = np.sqrt((x - center_x)**2 + y**2) / np.sqrt(center_x**2 + h**2)
                depth_map[y, x] += distance_factor * 0.3
        
        depth_map = np.clip(depth_map, 0, 1)
        
        # Apply atmospheric scattering model
        beta = np.random.uniform(1.5, 3.5)  # Scattering coefficient
        transmission = np.exp(-beta * depth_map)
        transmission = np.maximum(transmission, 0.05)
        transmission = transmission[:, :, np.newaxis]
        
        # Atmospheric light with slight color variation
        atmospheric_light = np.random.uniform(0.8, 1.0, 3)
        atmospheric_light[2] *= 0.95  # Slightly blue atmospheric light
        
        # Apply scattering: I = J * t + A * (1 - t)
        hazy_img = clear_img * transmission + atmospheric_light * (1 - transmission)
        
        # Add slight color shift in hazy conditions
        color_shift = np.random.uniform(0.95, 1.05, 3)
        hazy_img = hazy_img * color_shift
        
        return np.clip(hazy_img, 0, 1)

class CombinedLoss(nn.Module):
    """Advanced loss function combining multiple objectives"""
    
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        
    def forward(self, pred, target):
        # L1 loss for overall structure
        l1 = self.l1_loss(pred, target)
        
        # MSE loss for fine details
        mse = self.mse_loss(pred, target)
        
        # Perceptual loss (simplified)
        pred_gray = torch.mean(pred, dim=1, keepdim=True)
        target_gray = torch.mean(target, dim=1, keepdim=True)
        
        # Edge preservation loss
        pred_edges = self.sobel_edges(pred_gray)
        target_edges = self.sobel_edges(target_gray)
        edge_loss = self.l1_loss(pred_edges, target_edges)
        
        # Color consistency loss
        pred_mean = torch.mean(pred, dim=[2, 3])
        target_mean = torch.mean(target, dim=[2, 3])
        color_loss = self.mse_loss(pred_mean, target_mean)
        
        # Combined loss
        total_loss = l1 + 0.1 * mse + 0.2 * edge_loss + 0.1 * color_loss
        
        return total_loss
    
    def sobel_edges(self, img):
        """Compute Sobel edges"""
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        
        sobel_x = sobel_x.view(1, 1, 3, 3).to(img.device)
        sobel_y = sobel_y.view(1, 1, 3, 3).to(img.device)
        
        edge_x = F.conv2d(img, sobel_x, padding=1)
        edge_y = F.conv2d(img, sobel_y, padding=1)
        
        return torch.sqrt(edge_x**2 + edge_y**2)

def train_superior_model():
    """Train the superior dehazing model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Training on device: {device}")
    
    # Create model
    model = SuperiorDehazingNet().to(device)
    
    # Create dataset
    train_dataset = AdvancedSyntheticDataset(num_samples=2000)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    
    # Loss and optimizer
    criterion = CombinedLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    
    # Training parameters
    num_epochs = 30
    best_loss = float('inf')
    
    logger.info("Starting superior model training...")
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for batch_idx, (hazy, clear) in enumerate(progress_bar):
            hazy, clear = hazy.to(device), clear.to(device)
            
            optimizer.zero_grad()
            output = model(hazy)
            loss = criterion(output, clear)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        avg_loss = epoch_loss / len(train_loader)
        logger.info(f'Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}')
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, 'superior_dehazing_model.pth')
            logger.info(f'✅ New best model saved with loss: {avg_loss:.4f}')
        
        scheduler.step()
    
    logger.info("🎉 Superior model training completed!")
    return model

def test_superior_model():
    """Test the superior trained model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = SuperiorDehazingNet().to(device)
    
    if os.path.exists('superior_dehazing_model.pth'):
        checkpoint = torch.load('superior_dehazing_model.pth', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info("✅ Loaded superior trained model")
    else:
        logger.error("❌ No trained model found! Please train first.")
        return None
    
    model.eval()
    
    # Test on sample image
    test_image_path = "test_hazy_image.jpg"
    if not os.path.exists(test_image_path):
        logger.error(f"❌ Test image {test_image_path} not found!")
        return None
    
    # Load and preprocess image
    img = cv2.imread(test_image_path)
    img = cv2.resize(img, (256, 256))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
    img_tensor = torch.FloatTensor(img_rgb).permute(2, 0, 1).unsqueeze(0).to(device)
    
    # Process with model
    with torch.no_grad():
        output = model(img_tensor)
        output_img = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
        output_img = np.clip(output_img * 255, 0, 255).astype(np.uint8)
        output_img = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
    
    # Save result
    output_path = 'superior_model_result.jpg'
    cv2.imwrite(output_path, output_img)
    logger.info(f"✅ Superior model result saved as '{output_path}'")
    
    # Calculate improvement metrics
    original_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    result_gray = cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)
    
    orig_contrast = np.std(original_gray) / 255.0
    result_contrast = np.std(result_gray) / 255.0
    
    orig_dark_channel = np.mean(np.min(img, axis=2)) / 255.0
    result_dark_channel = np.mean(np.min(output_img, axis=2)) / 255.0
    
    logger.info(f"📊 Performance Metrics:")
    logger.info(f"   Contrast: {orig_contrast:.3f} → {result_contrast:.3f} (Δ{result_contrast-orig_contrast:+.3f})")
    logger.info(f"   Haze Level: {orig_dark_channel:.3f} → {result_dark_channel:.3f} (Δ{orig_dark_channel-result_dark_channel:+.3f})")
    
    return output_path

if __name__ == "__main__":
    print("🧠 Superior Dehazing Model Training System")
    print("=" * 50)
    
    choice = input("Choose option:\n1. Train superior model\n2. Test existing model\n3. Both\nEnter choice (1/2/3): ")
    
    if choice in ['1', '3']:
        print("\n🚀 Starting superior model training...")
        train_superior_model()
    
    if choice in ['2', '3']:
        print("\n🧪 Testing superior trained model...")
        result = test_superior_model()
        if result:
            print(f"✅ Results saved to: {result}")
    
    print("\n🎉 Process completed!")
