"""
Final Perfect Balanced Training
==============================

A final optimized training approach that achieves the perfect balance quickly and efficiently.
This script will train a model that provides crystal clear results while maintaining natural appearance.
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

class FinalPerfectNet(nn.Module):
    """Final Perfect Balanced Dehazing Network - Optimized and Stable"""
    
    def __init__(self):
        super(FinalPerfectNet, self).__init__()
        
        # Simplified but effective encoder
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(128, 32, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 128, 1),
            nn.Sigmoid()
        )
        
        # Decoder
        self.conv4 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv5 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv6 = nn.Conv2d(32, 3, 3, padding=1)
        
        # Activation functions
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Store input for residual connection
        input_img = x
        
        # Encoder
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x1))
        x3 = self.relu(self.conv3(x2))
        
        # Attention
        attention_weights = self.attention(x3)
        x3_attended = x3 * attention_weights
        
        # Decoder
        x4 = self.relu(self.conv4(x3_attended))
        x5 = self.relu(self.conv5(x4))
        dehazed = self.sigmoid(self.conv6(x5))
        
        # Perfect balance: 75% dehazed + 25% original for crystal clarity with naturalness
        balanced = dehazed * 0.75 + input_img * 0.25
        
        return torch.clamp(balanced, 0, 1)

class FinalDataset(Dataset):
    """Final optimized dataset"""
    
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
        
        # Repeat pairs for more training data
        self.pairs = self.pairs * 5  # 5x augmentation
        
        logger.info(f"Found {len(self.pairs)} training samples")
    
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
        
        # Random augmentation
        if np.random.random() > 0.5:
            # Brightness adjustment
            factor = np.random.uniform(0.8, 1.2)
            hazy = np.clip(hazy * factor, 0, 255)
            clear = np.clip(clear * factor, 0, 255)
        
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

class FinalBalancedLoss(nn.Module):
    """Final balanced loss for optimal results"""
    
    def __init__(self):
        super(FinalBalancedLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
    
    def forward(self, pred, target):
        # Basic reconstruction
        mse_loss = self.mse(pred, target)
        l1_loss = self.l1(pred, target)
        
        # Color consistency
        pred_mean = torch.mean(pred, dim=(2, 3))
        target_mean = torch.mean(target, dim=(2, 3))
        color_loss = torch.mean(torch.abs(pred_mean - target_mean))
        
        # Balanced combination for perfect results
        total_loss = 0.4 * mse_loss + 0.4 * l1_loss + 0.2 * color_loss
        
        return total_loss

def train_final_model():
    """Train the final perfect balanced model"""
    
    logger.info("Starting Final Perfect Balanced Training")
    logger.info("="*60)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Model
    model = FinalPerfectNet().to(device)
    logger.info("Final Perfect Balanced Network created")
    
    # Dataset
    dataset = FinalDataset('data/train/hazy', 'data/train/clear')
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # Loss and optimizer
    criterion = FinalBalancedLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # Training
    num_epochs = 30  # Reduced for efficiency
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
        
        avg_loss = total_loss / len(dataloader)
        
        # Save best model
        if avg_loss < best_loss and not np.isnan(avg_loss):
            best_loss = avg_loss
            os.makedirs('models/final_perfect_balanced', exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'loss': best_loss,
                'epoch': epoch
            }, 'models/final_perfect_balanced/final_perfect_model.pth')
        
        if epoch % 5 == 0:
            logger.info(f"Epoch {epoch+1}: Loss = {avg_loss:.6f} (Best: {best_loss:.6f})")
    
    logger.info("Training completed!")
    logger.info(f"Best Loss: {best_loss:.6f}")
    
    return model

def test_final_model():
    """Test the final model"""
    
    logger.info("Testing final model...")
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FinalPerfectNet().to(device)
    
    model_path = 'models/final_perfect_balanced/final_perfect_model.pth'
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
        output_name = os.path.basename(img_path).replace('.jpg', '_final_perfect_balanced.jpg')
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
    
    # Train the final model
    model = train_final_model()
    
    # Test the model
    test_final_model()
    
    logger.info("Final Perfect Balanced Training Complete!")
    logger.info("Check test_results/ folder for final output images")

if __name__ == "__main__":
    main()
