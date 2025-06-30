"""
Quick Perfect Balanced Training
==============================

A simplified training approach to quickly get a working perfectly balanced model
that matches your playground reference image quality.
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

class SimplePerfectNet(nn.Module):
    """Simplified Perfect Balanced Dehazing Network"""
    
    def __init__(self):
        super(SimplePerfectNet, self).__init__()
        
        # Simple encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        
        # Attention for balanced processing
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(128, 32, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 128, 1),
            nn.Sigmoid()
        )
        
        # Simple decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Store input for residual connection
        input_img = x
        
        # Encoder
        encoded = self.encoder(x)
        
        # Attention
        attention_weights = self.attention(encoded)
        attended = encoded * attention_weights
        
        # Decoder
        decoded = self.decoder(attended)
        
        # Perfect balance: 60% dehazed + 40% original for natural appearance
        balanced = decoded * 0.6 + input_img * 0.4
        
        return torch.clamp(balanced, 0, 1)

class SimpleDataset(Dataset):
    """Simple dataset for quick training"""
    
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
        
        logger.info(f"Found {len(self.pairs)} image pairs")
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        filename = self.pairs[idx]
        
        # Load images
        hazy = cv2.imread(os.path.join(self.hazy_dir, filename))
        clear = cv2.imread(os.path.join(self.clear_dir, filename))
        
        # Convert BGR to RGB
        hazy = cv2.cvtColor(hazy, cv2.COLOR_BGR2RGB)
        clear = cv2.cvtColor(clear, cv2.COLOR_BGR2RGB)
        
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

class PerfectBalancedLoss(nn.Module):
    """Perfect balanced loss for natural results"""
    
    def __init__(self):
        super(PerfectBalancedLoss, self).__init__()
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
        
        # Perfect balance combination
        total_loss = 0.5 * mse_loss + 0.3 * l1_loss + 0.2 * color_loss
        
        return total_loss

def train_quick_model():
    """Train the model quickly"""
    
    logger.info("Starting Quick Perfect Balanced Training")
    logger.info("="*50)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Model
    model = SimplePerfectNet().to(device)
    
    # Dataset
    dataset = SimpleDataset('data/train/hazy', 'data/train/clear')
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # Loss and optimizer
    criterion = PerfectBalancedLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training
    num_epochs = 50
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
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            os.makedirs('models/quick_perfect_balanced', exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'loss': best_loss,
                'epoch': epoch
            }, 'models/quick_perfect_balanced/quick_perfect_model.pth')
        
        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch+1}: Loss = {avg_loss:.6f} (Best: {best_loss:.6f})")
    
    logger.info("Training completed!")
    logger.info(f"Best Loss: {best_loss:.6f}")
    
    return model

def test_on_playground():
    """Test the trained model on playground image"""
    
    logger.info("Testing on playground image...")
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimplePerfectNet().to(device)
    
    model_path = 'models/quick_perfect_balanced/quick_perfect_model.pth'
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
        output_name = os.path.basename(img_path).replace('.jpg', '_quick_perfect_balanced.jpg')
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
    """Main training and testing function"""
    
    # Train the model
    model = train_quick_model()
    
    # Test on playground image
    test_on_playground()
    
    logger.info("Quick Perfect Balanced Training Complete!")
    logger.info("Check test_results/ folder for output images")

if __name__ == "__main__":
    main()
