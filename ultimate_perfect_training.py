"""
Ultimate Perfect Balanced Training
=================================

The ultimate training approach that achieves the perfect balance:
- Strong clarity improvement (crystal clear visibility)
- Natural appearance (not too aggressive)
- Professional quality matching your reference image

This model will provide significant clarity improvement while maintaining natural colors.
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

class UltimatePerfectNet(nn.Module):
    """Ultimate Perfect Balanced Dehazing Network"""
    
    def __init__(self):
        super(UltimatePerfectNet, self).__init__()
        
        # Enhanced encoder for better feature extraction
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        
        # Enhanced attention mechanism
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(256, 64, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 256, 1),
            nn.Sigmoid()
        )
        
        # Decoder with strong clarity enhancement
        self.conv4 = nn.Conv2d(256, 128, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 64, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(64)
        self.conv6 = nn.Conv2d(64, 32, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(32)
        self.conv7 = nn.Conv2d(32, 3, 3, padding=1)
        
        # Clarity enhancement layer
        self.clarity_enhance = nn.Conv2d(3, 3, 1)
        
        # Activation functions
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Store input for residual connection
        input_img = x
        
        # Enhanced encoder
        x1 = self.relu(self.bn1(self.conv1(x)))
        x2 = self.relu(self.bn2(self.conv2(x1)))
        x3 = self.relu(self.bn3(self.conv3(x2)))
        
        # Enhanced attention
        attention_weights = self.attention(x3)
        x3_attended = x3 * attention_weights
        
        # Enhanced decoder
        x4 = self.relu(self.bn4(self.conv4(x3_attended)))
        x5 = self.relu(self.bn5(self.conv5(x4)))
        x6 = self.relu(self.bn6(self.conv6(x5)))
        dehazed_raw = self.sigmoid(self.conv7(x6))
        
        # Clarity enhancement
        dehazed_enhanced = self.sigmoid(self.clarity_enhance(dehazed_raw))
        
        # Ultimate balance: Strong dehazing with natural preservation
        # 85% enhanced dehazed + 15% original for crystal clarity with naturalness
        balanced = dehazed_enhanced * 0.85 + input_img * 0.15
        
        return torch.clamp(balanced, 0, 1)

class UltimateDataset(Dataset):
    """Ultimate dataset with strong augmentation"""
    
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
        
        # Strong augmentation - repeat pairs multiple times
        self.pairs = self.pairs * 10  # 10x augmentation for better training
        
        logger.info(f"Found {len(self.pairs)} training samples with strong augmentation")
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        filename = self.pairs[idx % len(self.pairs)]
        
        # Load images
        hazy = cv2.imread(os.path.join(self.hazy_dir, filename))
        clear = cv2.imread(os.path.join(self.clear_dir, filename))
        
        # Convert BGR to RGB
        hazy = cv2.cvtColor(hazy, cv2.COLOR_BGR2RGB)
        clear = cv2.cvtColor(clear, cv2.COLOR_BGR2RGB)
        
        # Strong augmentation for better clarity learning
        aug_type = np.random.randint(0, 4)
        
        if aug_type == 0:
            # Brightness variation
            factor = np.random.uniform(0.7, 1.3)
            hazy = np.clip(hazy * factor, 0, 255)
            clear = np.clip(clear * factor, 0, 255)
        elif aug_type == 1:
            # Contrast variation
            factor = np.random.uniform(0.8, 1.2)
            hazy = np.clip((hazy - 128) * factor + 128, 0, 255)
            clear = np.clip((clear - 128) * factor + 128, 0, 255)
        elif aug_type == 2:
            # Saturation variation
            hsv_hazy = cv2.cvtColor(hazy, cv2.COLOR_RGB2HSV)
            hsv_clear = cv2.cvtColor(clear, cv2.COLOR_RGB2HSV)
            factor = np.random.uniform(0.8, 1.2)
            hsv_hazy[:, :, 1] = np.clip(hsv_hazy[:, :, 1] * factor, 0, 255)
            hsv_clear[:, :, 1] = np.clip(hsv_clear[:, :, 1] * factor, 0, 255)
            hazy = cv2.cvtColor(hsv_hazy, cv2.COLOR_HSV2RGB)
            clear = cv2.cvtColor(hsv_clear, cv2.COLOR_HSV2RGB)
        # else: no augmentation
        
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

class UltimateBalancedLoss(nn.Module):
    """Ultimate balanced loss for maximum clarity with naturalness"""
    
    def __init__(self):
        super(UltimateBalancedLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
    
    def forward(self, pred, target):
        # Basic reconstruction
        mse_loss = self.mse(pred, target)
        l1_loss = self.l1(pred, target)
        
        # Perceptual loss for better visual quality
        pred_gray = 0.299 * pred[:, 0] + 0.587 * pred[:, 1] + 0.114 * pred[:, 2]
        target_gray = 0.299 * target[:, 0] + 0.587 * target[:, 1] + 0.114 * target[:, 2]
        perceptual_loss = self.l1(pred_gray, target_gray)
        
        # Color consistency
        pred_mean = torch.mean(pred, dim=(2, 3))
        target_mean = torch.mean(target, dim=(2, 3))
        color_loss = torch.mean(torch.abs(pred_mean - target_mean))
        
        # Clarity enhancement loss
        pred_edges = self.calculate_edges(pred)
        target_edges = self.calculate_edges(target)
        clarity_loss = self.l1(pred_edges, target_edges)
        
        # Ultimate balanced combination for maximum clarity
        total_loss = (
            0.25 * mse_loss +        # Basic reconstruction
            0.25 * l1_loss +         # Color preservation
            0.2 * perceptual_loss +  # Visual quality
            0.15 * color_loss +      # Color consistency
            0.15 * clarity_loss      # Clarity enhancement
        )
        
        return total_loss
    
    def calculate_edges(self, images):
        """Calculate edge maps for clarity enhancement"""
        # Convert to grayscale
        gray = 0.299 * images[:, 0:1] + 0.587 * images[:, 1:2] + 0.114 * images[:, 2:3]
        
        # Simple edge detection using gradient
        grad_x = torch.abs(gray[:, :, :, 1:] - gray[:, :, :, :-1])
        grad_y = torch.abs(gray[:, :, 1:, :] - gray[:, :, :-1, :])
        
        # Pad to maintain size
        grad_x = torch.nn.functional.pad(grad_x, (0, 1, 0, 0))
        grad_y = torch.nn.functional.pad(grad_y, (0, 0, 0, 1))
        
        edges = torch.sqrt(grad_x**2 + grad_y**2)
        return edges

def train_ultimate_model():
    """Train the ultimate perfect balanced model"""
    
    logger.info("Starting Ultimate Perfect Balanced Training")
    logger.info("="*60)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Model
    model = UltimatePerfectNet().to(device)
    logger.info("Ultimate Perfect Balanced Network created")
    
    # Dataset
    dataset = UltimateDataset('data/train/hazy', 'data/train/clear')
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # Loss and optimizer
    criterion = UltimateBalancedLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.7)
    
    # Training
    num_epochs = 40
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        
        for hazy, clear in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            hazy, clear = hazy.to(device), clear.to(device)
            
            optimizer.zero_grad()
            dehazed = model(hazy)
            loss = criterion(dehazed, clear)
            
            # Check for NaN
            if torch.isnan(loss):
                logger.warning("NaN loss detected, skipping batch")
                continue
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item()
        
        scheduler.step()
        avg_loss = total_loss / len(dataloader)
        
        # Save best model
        if avg_loss < best_loss and not np.isnan(avg_loss):
            best_loss = avg_loss
            os.makedirs('models/ultimate_perfect_balanced', exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'loss': best_loss,
                'epoch': epoch
            }, 'models/ultimate_perfect_balanced/ultimate_perfect_model.pth')
        
        if epoch % 5 == 0:
            logger.info(f"Epoch {epoch+1}: Loss = {avg_loss:.6f} (Best: {best_loss:.6f})")
    
    logger.info("Training completed!")
    logger.info(f"Best Loss: {best_loss:.6f}")
    
    return model

def test_ultimate_model():
    """Test the ultimate model"""
    
    logger.info("Testing ultimate model...")
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UltimatePerfectNet().to(device)
    
    model_path = 'models/ultimate_perfect_balanced/ultimate_perfect_model.pth'
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
        output_name = os.path.basename(img_path).replace('.jpg', '_ultimate_perfect_balanced.jpg')
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
    
    # Train the ultimate model
    model = train_ultimate_model()
    
    # Test the model
    test_ultimate_model()
    
    logger.info("Ultimate Perfect Balanced Training Complete!")
    logger.info("Check test_results/ folder for ultimate output images")

if __name__ == "__main__":
    main()
