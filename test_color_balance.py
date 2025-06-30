import os
import sys
import torch
from utils.model import AODNet, LightDehazeNet, create_model
import cv2
import numpy as np
from PIL import Image
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_dehazing_models():
    """Test both dehazing models with a sample image to verify color balance"""
    
    print("Starting dehazing model test with color balance verification")
    
    # Check CUDA availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Define models to test
    models_to_test = ['aod', 'enhanced', 'light']
    
    # Find test images in uploads folder
    uploads_dir = 'static/uploads'
    if not os.path.exists(uploads_dir):
        print(f"Uploads directory {uploads_dir} not found")
        return
    
    # Get first image file from uploads
    test_images = []
    for file in os.listdir(uploads_dir):
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
            test_images.append(os.path.join(uploads_dir, file))
            if len(test_images) >= 2:  # Limit to 2 test images
                break
    
    if not test_images:
        print("No test images found in uploads folder")
        return
    
    # Create results directory
    results_dir = 'test_results'
    os.makedirs(results_dir, exist_ok=True)
    
    # Test each model with each image
    for model_type in models_to_test:
        try:
            # Load model
            print(f"\nTesting {model_type} model...")
            model = create_model(model_type, device)
            model.eval()
            
            # Process each test image
            for img_path in test_images:
                img_name = os.path.basename(img_path)
                print(f"Processing {img_name} with {model_type} model")
                
                # Load and preprocess image
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Could not read image at {img_path}")
                    continue
                
                # Convert to RGB and normalize
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img.astype(np.float32) / 255.0
                
                # Convert to tensor
                img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(device)
                
                # Process with model
                with torch.no_grad():
                    output = model(img_tensor)
                
                # Convert output to numpy array
                output = output.squeeze().cpu().numpy().transpose(1, 2, 0)
                output = np.clip(output, 0, 1)
                
                # Convert to uint8
                output = (output * 255).astype(np.uint8)
                
                # Convert back to BGR for saving with OpenCV
                output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
                
                # Save result
                output_path = os.path.join(results_dir, f"{model_type}_{img_name}")
                cv2.imwrite(output_path, output)
                print(f"Saved result to {output_path}")
                
                # Analyze color balance
                # Calculate mean RGB values for input and output to verify color balance
                input_img = cv2.imread(img_path)
                input_rgb_means = cv2.mean(input_img)[:3]  # BGR order
                output_rgb_means = cv2.mean(output)[:3]    # BGR order
                
                print(f"Input image RGB means (BGR order): {input_rgb_means}")
                print(f"Output image RGB means (BGR order): {output_rgb_means}")
                
                # Check if red channel is dominant in output
                if output_rgb_means[2] > 1.5 * output_rgb_means[0] or output_rgb_means[2] > 1.5 * output_rgb_means[1]:
                    print("WARNING: Red channel still appears dominant in output")
                else:
                    print("Color balance looks good - no red tint detected")
                
        except Exception as e:
            print(f"Error testing {model_type} model: {str(e)}")
    
    print("\nTesting completed! Results saved to", results_dir)
    
if __name__ == "__main__":
    test_dehazing_models()
