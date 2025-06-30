import torch
import os
from utils.model import DeepDehazeNet

# Create the directory for weights
weights_dir = 'static/models/weights'
os.makedirs(weights_dir, exist_ok=True)

# Path for deep model weights
deep_weights_path = os.path.join(weights_dir, 'deep_net.pth')

print("Creating DeepDehazeNet model...")
model = DeepDehazeNet()

print("Initializing model parameters...")
# Initialize using standard initialization
for m in model.modules():
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.01)

# Special weight initialization for transmission map estimation
if hasattr(model, 'trans_conv2'):
    torch.nn.init.normal_(model.trans_conv2.weight.data, mean=0.0, std=0.01)
    if model.trans_conv2.bias is not None:
        torch.nn.init.constant_(model.trans_conv2.bias.data, 0.3)

print(f"Saving model weights to {deep_weights_path}")
torch.save(model.state_dict(), deep_weights_path)
print("Model weights saved successfully!")

# Verify file size
if os.path.exists(deep_weights_path):
    size_kb = os.path.getsize(deep_weights_path) / 1024
    print(f"Weight file size: {size_kb:.1f} KB")
    print("DeepDehazeNet weights generation completed successfully.")
else:
    print("Error: Weight file was not created.")
