import torch
import os
from utils.model import load_model
import cv2
import numpy as np

# Configure device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def test_deep_model():
    """Test the DeepDehazeNet model with a simple image"""
    # Create a test input
    test_img = np.ones((256, 256, 3), dtype=np.float32)  # Simple test image
    test_img = test_img * 0.5  # Set to mid-gray
    
    # Convert to tensor [C, H, W]
    img_tensor = torch.from_numpy(test_img.transpose(2, 0, 1)).unsqueeze(0).to(device)
    
    print("Loading DeepDehazeNet model...")
    model = load_model(device, 'deep')
    
    # Set model to evaluation mode
    model.eval()
    
    print("Running inference on test image...")
    with torch.no_grad():
        try:
            output = model(img_tensor)
            print(f"Success! Output shape: {output.shape}")
            
            # Convert output to numpy and save
            output_np = output.squeeze().cpu().numpy().transpose(1, 2, 0)
            output_np = np.clip(output_np, 0, 1) * 255
            output_np = output_np.astype(np.uint8)
            
            # Save the result
            os.makedirs('test_results', exist_ok=True)
            cv2.imwrite('test_results/deep_model_test.jpg', cv2.cvtColor(output_np, cv2.COLOR_RGB2BGR))
            print("Saved test output to test_results/deep_model_test.jpg")
            return True
            
        except Exception as e:
            print(f"Error during inference: {str(e)}")
            return False

if __name__ == "__main__":
    success = test_deep_model()
    if success:
        print("DeepDehazeNet model test PASSED!")
    else:
        print("DeepDehazeNet model test FAILED!")
