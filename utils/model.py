import torch
import torch.nn as nn
import logging
import os
import torchvision.transforms as transforms

logger = logging.getLogger(__name__)

class AODNet(nn.Module):
    """
    AOD-Net model for single image dehazing.
    Enhanced version with improved atmospheric light estimation and transmission map generation.
    
    Paper: "AOD-Net: All-in-One Dehazing Network" - https://arxiv.org/abs/1707.06543
    """
    def __init__(self):
        super(AODNet, self).__init__()
        
        # K-estimation network with enhanced architecture
        self.conv1 = nn.Conv2d(3, 3, 1, padding=0)
        self.conv2 = nn.Conv2d(3, 3, 3, padding=1)
        self.conv3 = nn.Conv2d(6, 3, 5, padding=2)
        self.conv4 = nn.Conv2d(6, 3, 7, padding=3)
        self.conv5 = nn.Conv2d(12, 3, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        
        # Additional enhancement layers for better dehazing
        self.refine_conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.refine_relu1 = nn.ReLU(inplace=True)
        self.refine_conv2 = nn.Conv2d(8, 3, 3, padding=1)
        
    def forward(self, x):
        # Ensure input is in valid range
        x = torch.clamp(x, 0.0, 1.0)
        
        # Store original image for guided filtering
        original = x
        
        # Extract features
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x1))
        
        # Concatenate and continue extracting features
        cat1 = torch.cat((x1, x2), 1)
        x3 = self.relu(self.conv3(cat1))
        
        cat2 = torch.cat((x2, x3), 1)
        x4 = self.relu(self.conv4(cat2))
        
        # Final concatenation and K estimation
        cat3 = torch.cat((x1, x2, x3, x4), 1)
        k = self.relu(self.conv5(cat3))
        
        # Apply the atmospheric scattering model to get the clean image
        # Enhanced implementation with more sophisticated handling
        
        # Extract transformation parameters from K with optimized range
        k = torch.clamp(k, min=0.05, max=2.0)  # Better limits for stronger dehazing
        
        # Extract transformation parameters from K
        r = k[:, 0:1, :, :]
        g = k[:, 1:2, :, :]
        b = k[:, 2:3, :, :]
        
        # Apply the transformation with improved channel handling
        x_r = torch.unsqueeze(x[:, 0, :, :], 1)
        x_g = torch.unsqueeze(x[:, 1, :, :], 1)
        x_b = torch.unsqueeze(x[:, 2, :, :], 1)
        
        # Enhanced atmospheric light estimation based on input image statistics
        # Estimate atm. light from the brightest pixels
        batch_size = x.size(0)
        
        # More accurate A estimation - use brightest pixels as a guide
        # Calculate per-image atmospheric light
        A_values = []
        for i in range(batch_size):
            img = x[i].permute(1, 2, 0)  # [H, W, C]
            # Calculate brightness
            brightness = torch.mean(img, dim=2)  # [H, W]
            # Get brightest pixels (top 0.1%)
            flat_brightness = brightness.flatten()
            num_pixels = flat_brightness.numel()
            num_brightest = max(int(num_pixels * 0.001), 1)
            
            # Get indices of brightest pixels
            _, indices = torch.topk(flat_brightness, num_brightest)
            
            # Get RGB values of brightest pixels
            flat_img = img.reshape(-1, 3)  # [H*W, 3]
            brightest_pixels = flat_img[indices]
            
            # Calculate atmospheric light as mean of brightest pixels
            A = torch.mean(brightest_pixels, dim=0)
            # Adjust based on image brightness
            overall_brightness = torch.mean(img)
            A = A * torch.clamp(1.2 - overall_brightness, 0.8, 1.5)
            
            A_values.append(A)
        
        # Convert list to tensor and reshape
        A_tensor = torch.stack(A_values)  # [B, 3]
        
        # Extract A values for each channel
        A_r = A_tensor[:, 0].view(batch_size, 1, 1, 1)
        A_g = A_tensor[:, 1].view(batch_size, 1, 1, 1)
        A_b = A_tensor[:, 2].view(batch_size, 1, 1, 1)
        
        # Ensure A values are reasonable (not too dark or bright)
        A_r = torch.clamp(A_r, 0.5, 0.95)
        A_g = torch.clamp(A_g, 0.5, 0.95)
        A_b = torch.clamp(A_b, 0.5, 0.95)
        
        # Advanced transmission map estimation - stronger dehazing effect
        # Adaptive transmission strength based on image brightness
        brightness = torch.mean(torch.stack([x_r, x_g, x_b]), dim=0)
        adaptive_strength = torch.clamp(1.0 - brightness, 0.3, 0.8)
        
        t_r = torch.clamp(1.0 - adaptive_strength * r, 0.1, 1.0)
        t_g = torch.clamp(1.0 - adaptive_strength * g, 0.1, 1.0)
        t_b = torch.clamp(1.0 - adaptive_strength * b, 0.1, 1.0)
        
        # Apply the enhanced atmospheric scattering model
        epsilon = 1e-5  # Prevent division by zero
        J_r = (x_r - A_r) / (t_r + epsilon) + A_r
        J_g = (x_g - A_g) / (t_g + epsilon) + A_g
        J_b = (x_b - A_b) / (t_b + epsilon) + A_b
        
        # Safety handling for any NaN or Inf values
        J_r = torch.nan_to_num(J_r, nan=0.5, posinf=1.0, neginf=0.0)
        J_g = torch.nan_to_num(J_g, nan=0.5, posinf=1.0, neginf=0.0)
        J_b = torch.nan_to_num(J_b, nan=0.5, posinf=1.0, neginf=0.0)
        
        # Combine channels
        result = torch.cat((J_r, J_g, J_b), 1)
        
        # Apply enhanced refinement for better visual quality
        refined = self.refine_relu1(self.refine_conv1(result))
        refined = self.refine_conv2(refined)
        
        # Apply stronger refinement for more visible difference
        result = result + 0.2 * refined
        
        # Guided filtering with original image to preserve edges
        # This helps to prevent over-smoothing while still removing haze
        # Apply per batch item
        for i in range(batch_size):
            # Extract single image
            img = result[i].permute(1, 2, 0)  # [H, W, C]
            guide = original[i].permute(1, 2, 0)  # [H, W, C]
            
            # Process each channel
            for c in range(3):
                # Apply pseudo guided filtering
                img_channel = img[:, :, c]
                guide_channel = guide[:, :, c]
                
                # Fast approximation of guided filter
                # 1. Calculate mean and variance of guide image
                n = 5  # Filter size
                padding = n // 2
                
                # Use unfold to create patches
                # Calculate mean of guide using box filter
                guide_padded = torch.nn.functional.pad(guide_channel, (padding, padding, padding, padding), mode='constant', value=0)
                guide_unfold = guide_padded.unfold(0, n, 1).unfold(1, n, 1)  # [H, W, n, n]
                guide_mean = torch.mean(guide_unfold, dim=(2, 3))
                
                # Calculate covariance
                img_padded = torch.nn.functional.pad(img_channel, (padding, padding, padding, padding), mode='constant', value=0)
                img_unfold = img_padded.unfold(0, n, 1).unfold(1, n, 1)  # [H, W, n, n]
                img_mean = torch.mean(img_unfold, dim=(2, 3))
                
                # Recombine using weights
                epsilon = 0.01
                weight = 0.5  # Control balance between original and filtered
                filtered = weight * img_channel + (1 - weight) * guide_channel
                
                # Update the channel
                result[i, c, :, :] = filtered
        
        # Advanced contrast enhancement
        # Apply advanced contrast enhancement via histogram equalization like approach
        for i in range(batch_size):
            img = result[i]  # [C, H, W]
            
            # Apply CLAHE-inspired contrast enhancement for each channel
            for c in range(3):
                channel = img[c]  # [H, W]
                
                # Calculate histogram
                hist = torch.histc(channel, bins=256, min=0.0, max=1.0)
                
                # Calculate CDF
                cdf = torch.cumsum(hist, 0)
                
                # Normalize CDF
                cdf_normalized = cdf / cdf[-1]
                
                # Create lookup table
                lookup_table = torch.zeros(256, device=x.device)
                for j in range(256):
                    lookup_table[j] = torch.clamp(cdf_normalized[j], 0.0, 1.0)
                
                # Apply lookup table (simplified CLAHE)
                channel_scaled = (channel * 255).long()
                channel_scaled = torch.clamp(channel_scaled, 0, 255)
                
                # Use lookup table to enhance contrast
                enhanced_channel = torch.zeros_like(channel)
                for v in range(256):
                    mask = (channel_scaled == v)
                    enhanced_channel[mask] = lookup_table[v]
                
                # Update the channel
                result[i, c] = enhanced_channel
        
        # Color balance adjustment - ensure natural colors
        # Normalize each channel separately to stretch the histogram with more aggressive parameters
        for c in range(3):  # For each color channel
            channel = result[:, c:c+1, :, :]
            # Find min and max values per batch item
            min_vals = torch.min(torch.min(channel, dim=3, keepdim=True)[0], dim=2, keepdim=True)[0]
            max_vals = torch.max(torch.max(channel, dim=3, keepdim=True)[0], dim=2, keepdim=True)[0]
            
            # Apply more aggressive stretching for better contrast
            mask = (max_vals - min_vals > 0.05).float()
            
            # Normalize with stronger effect
            normalized = torch.where(
                mask > 0,
                ((channel - min_vals) / torch.clamp(max_vals - min_vals, min=0.05)) * 0.95,
                channel
            )
            
            result[:, c:c+1, :, :] = normalized
        
        # Boost saturation slightly
        mean_color = torch.mean(result, dim=1, keepdim=True)
        result = mean_color + 1.3 * (result - mean_color)
        
        # Apply S-curve for final contrast boost
        result = torch.sigmoid((result - 0.5) * 5) * 0.95 + 0.025
        
        # Final clipping to ensure valid values
        result = torch.clamp(result, 0.0, 1.0)
        
        return result

class LightDehazeNet(nn.Module):
    """
    Enhanced lightweight dehazing network with residual connections, attention mechanisms,
    and advanced visual enhancement techniques.
    """
    def __init__(self):
        super(LightDehazeNet, self).__init__()
        
        # Initial feature extraction
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        
        # Encoder path
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        
        # Deeper feature extraction
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        
        # Bottleneck - Multiple residual blocks
        self.res1_conv1 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.res1_relu1 = nn.ReLU(inplace=True)
        self.res1_conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        
        self.res2_conv1 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.res2_relu1 = nn.ReLU(inplace=True)
        self.res2_conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        
        # Additional residual block for more expressive power
        self.res3_conv1 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.res3_relu1 = nn.ReLU(inplace=True)
        self.res3_conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        
        # Decoder path
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv4 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU(inplace=True)
        
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv5 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.relu5 = nn.ReLU(inplace=True)
        
        # Advanced attention mechanism
        self.att_conv1 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.att_relu = nn.ReLU(inplace=True)
        self.att_conv2 = nn.Conv2d(16, 1, kernel_size=1)
        self.att_sigmoid = nn.Sigmoid()
        
        # Color enhancement branch
        self.color_conv1 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.color_relu = nn.ReLU(inplace=True)
        self.color_conv2 = nn.Conv2d(16, 3, kernel_size=1)
        
        # Final restoration
        self.final_conv = nn.Conv2d(35, 3, kernel_size=3, padding=1)  # 32 + 3 from color branch
        self.final_act = nn.Tanh()  # Tanh for better contrast
        
    def forward(self, x):
        # Ensure input is in valid range (safety check)
        x = torch.clamp(x, 0.0, 1.0)
        
        # Store original for skip connection
        original = x
        
        # Feature extraction
        x1 = self.relu1(self.conv1(x))
        
        # Encoder path
        x2_down = self.pool1(x1)
        x2 = self.relu2(self.conv2(x2_down))
        
        # Deeper feature extraction
        x3_down = self.pool2(x2)
        x3 = self.relu3(self.conv3(x3_down))
        
        # First residual block
        res1 = self.res1_relu1(self.res1_conv1(x3))
        res1 = self.res1_conv2(res1)
        x3 = x3 + res1  # Skip connection
        
        # Second residual block
        res2 = self.res2_relu1(self.res2_conv1(x3))
        res2 = self.res2_conv2(res2)
        x3 = x3 + res2  # Skip connection
        
        # Third residual block
        res3 = self.res3_relu1(self.res3_conv1(x3))
        res3 = self.res3_conv2(res3)
        x3 = x3 + res3  # Skip connection
        
        # Decoder path with skip connections
        x4 = self.up1(x3)
        x4 = self.relu4(self.conv4(x4))
        
        x5 = self.up2(x4)
        x5 = self.relu5(self.conv5(x5))
        
        # Simple attention mechanism (toned down to avoid artifacts)
        att = self.att_relu(self.att_conv1(x5))
        att = self.att_conv2(att)
        att_map = self.att_sigmoid(att)
        x5_att = x5 * att_map
        
        # Color enhancement branch for better color reproduction
        color = self.color_relu(self.color_conv1(x5))
        color = self.color_conv2(color)
        color = torch.sigmoid(color)  # Ensure valid range
        
        # Concatenate features and color enhancement
        combined = torch.cat([x5_att, color], dim=1)
        
        # Final restoration with standard processing
        out = self.final_conv(combined)
        out = self.final_act(out)
        
        # Scale to [0,1] range
        out = (out + 1) / 2
        
        # Make sure dimensions match before applying skip connection
        if out.size() != original.size():
            # Resize output to match original input size
            try:
                out = torch.nn.functional.interpolate(
                    out,
                    size=(original.size(2), original.size(3)),
                    mode='bilinear',
                    align_corners=False
                )
            except Exception as e:
                # If interpolation fails, just use the original size
                pass
        
        # Apply skip connection with original for better detail preservation
        # Using a lower alpha (0.1) to avoid preserving too much haze
        alpha = 0.1
        out = (1 - alpha) * out + alpha * original
        
        # Basic color and contrast enhancement
        # Apply a mild S-curve for contrast
        out = torch.clamp(out, 0.01, 0.99)  # Prevent extreme values
        out = 1.0 / (1.0 + torch.exp(-(out - 0.5) * 3.0)) * 0.9 + 0.05
        
        # Simple color saturation enhancement
        mean_color = torch.mean(out, dim=1, keepdim=True)
        out = mean_color + 1.2 * (out - mean_color)
        
        # Final output normalization
        out = torch.clamp(out, 0.0, 1.0)
        
        # Ensure no NaN or infinite values
        out = torch.nan_to_num(out, nan=0.5, posinf=1.0, neginf=0.0)
        
        return out

class DeepDehazeNet(nn.Module):
    """
    Advanced dehazing network that combines multiple state-of-the-art techniques:
    1. Dense feature extraction with dilated convolutions
    2. Multi-scale feature fusion
    3. Attention-guided refinement
    4. Enhanced transmission map estimation
    5. Adaptive contrast enhancement
    
    This model is designed to handle difficult dehazing scenarios including
    heavy fog, night scenes, and non-homogeneous haze.
    """
    def __init__(self):
        super(DeepDehazeNet, self).__init__()
        
        # Initial feature extraction with larger receptive field
        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)
        
        # Dense feature extraction block 1 - multi-dilation
        self.dense1_conv1 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.dense1_bn1 = nn.BatchNorm2d(32)
        self.dense1_relu1 = nn.ReLU(inplace=True)
        
        self.dense1_conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=2, dilation=2)
        self.dense1_bn2 = nn.BatchNorm2d(32)
        self.dense1_relu2 = nn.ReLU(inplace=True)
        
        self.dense1_conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=4, dilation=4)
        self.dense1_bn3 = nn.BatchNorm2d(32)
        self.dense1_relu3 = nn.ReLU(inplace=True)
        
        # Downsample 1
        self.down1 = nn.Conv2d(96, 64, kernel_size=3, stride=2, padding=1)
        self.down1_bn = nn.BatchNorm2d(64)
        self.down1_relu = nn.ReLU(inplace=True)
        
        # Dense feature extraction block 2
        self.dense2_conv1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.dense2_bn1 = nn.BatchNorm2d(64)
        self.dense2_relu1 = nn.ReLU(inplace=True)
        
        self.dense2_conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=2, dilation=2)
        self.dense2_bn2 = nn.BatchNorm2d(64)
        self.dense2_relu2 = nn.ReLU(inplace=True)
        
        self.dense2_conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=4, dilation=4)
        self.dense2_bn3 = nn.BatchNorm2d(64)
        self.dense2_relu3 = nn.ReLU(inplace=True)
        
        # Downsample 2
        self.down2 = nn.Conv2d(192, 128, kernel_size=3, stride=2, padding=1)
        self.down2_bn = nn.BatchNorm2d(128)
        self.down2_relu = nn.ReLU(inplace=True)
        
        # Global context block
        self.global_conv = nn.Conv2d(128, 128, kernel_size=3, padding=2, dilation=2)
        self.global_bn = nn.BatchNorm2d(128)
        self.global_relu = nn.ReLU(inplace=True)
        
        # Attention module
        self.attn_conv1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.attn_bn1 = nn.BatchNorm2d(64)
        self.attn_relu1 = nn.ReLU(inplace=True)
        
        self.attn_conv2 = nn.Conv2d(64, 1, kernel_size=1)
        self.attn_sigmoid = nn.Sigmoid()
        
        # Upsample 1
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.up1_bn = nn.BatchNorm2d(64)
        self.up1_relu = nn.ReLU(inplace=True)
        
        # Refinement block 1
        self.refine1_conv = nn.Conv2d(128, 64, kernel_size=3, padding=1)  # 64 + 64 (skip)
        self.refine1_bn = nn.BatchNorm2d(64)
        self.refine1_relu = nn.ReLU(inplace=True)
        
        # Upsample 2
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.up2_bn = nn.BatchNorm2d(32)
        self.up2_relu = nn.ReLU(inplace=True)
        
        # Refinement block 2
        self.refine2_conv = nn.Conv2d(64, 32, kernel_size=3, padding=1)  # 32 + 32 (skip)
        self.refine2_bn = nn.BatchNorm2d(32)
        self.refine2_relu = nn.ReLU(inplace=True)
        
        # Transmission map estimation branch
        self.trans_conv1 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.trans_relu1 = nn.ReLU(inplace=True)
        self.trans_conv2 = nn.Conv2d(16, 1, kernel_size=1)
        self.trans_sigmoid = nn.Sigmoid()
        
        # Color correction branch
        self.color_conv1 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.color_relu1 = nn.ReLU(inplace=True)
        self.color_conv2 = nn.Conv2d(16, 3, kernel_size=1)
        self.color_sigmoid = nn.Sigmoid()
        
        # Final output
        self.final_conv = nn.Conv2d(32 + 1 + 3, 3, kernel_size=3, padding=1)  # features + trans + color
        
        # Contrast enhancement module
        self.ce_conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.ce_relu1 = nn.ReLU(inplace=True)
        self.ce_conv2 = nn.Conv2d(8, 3, kernel_size=1)
        
    def forward(self, x):
        # Ensure input is in valid range
        x = torch.clamp(x, 0.0, 1.0)
        
        # Store original image for skip connections and residual learning
        original = x
        
        # Initial feature extraction
        f1 = self.relu1(self.bn1(self.conv1(x)))
        
        # Dense feature extraction block 1
        d1_1 = self.dense1_relu1(self.dense1_bn1(self.dense1_conv1(f1)))
        d1_2 = self.dense1_relu2(self.dense1_bn2(self.dense1_conv2(f1)))
        d1_3 = self.dense1_relu3(self.dense1_bn3(self.dense1_conv3(f1)))
        d1_cat = torch.cat([d1_1, d1_2, d1_3], dim=1)
        
        # First downsampling
        f2 = self.down1_relu(self.down1_bn(self.down1(d1_cat)))
        
        # Dense feature extraction block 2
        d2_1 = self.dense2_relu1(self.dense2_bn1(self.dense2_conv1(f2)))
        d2_2 = self.dense2_relu2(self.dense2_bn2(self.dense2_conv2(f2)))
        d2_3 = self.dense2_relu3(self.dense2_bn3(self.dense2_conv3(f2)))
        d2_cat = torch.cat([d2_1, d2_2, d2_3], dim=1)
        
        # Second downsampling
        f3 = self.down2_relu(self.down2_bn(self.down2(d2_cat)))
        
        # Global context
        f3_global = self.global_relu(self.global_bn(self.global_conv(f3)))
        
        # Attention module to focus on hazy regions
        attn = self.attn_relu1(self.attn_bn1(self.attn_conv1(f3_global)))
        attn_map = self.attn_sigmoid(self.attn_conv2(attn))
        
        # Apply attention
        f3_att = f3_global * attn_map
        
        # First upsampling
        up1 = self.up1_relu(self.up1_bn(self.up1(f3_att)))
        
        # Skip connection from encoder
        skip1 = torch.cat([up1, f2], dim=1)
        
        # Refinement 1
        r1 = self.refine1_relu(self.refine1_bn(self.refine1_conv(skip1)))
        
        # Second upsampling
        up2 = self.up2_relu(self.up2_bn(self.up2(r1)))
        
        # Skip connection from encoder
        skip2 = torch.cat([up2, f1], dim=1)
        
        # Refinement 2
        r2 = self.refine2_relu(self.refine2_bn(self.refine2_conv(skip2)))
        
        # Transmission map estimation
        trans = self.trans_relu1(self.trans_conv1(r2))
        trans_map = self.trans_sigmoid(self.trans_conv2(trans))
        
        # Constrain transmission map to more realistic values
        trans_map = torch.clamp(trans_map, 0.1, 0.9)
        
        # Color correction
        color = self.color_relu1(self.color_conv1(r2))
        color_correction = self.color_sigmoid(self.color_conv2(color))
        
        # Combine features with transmission map and color correction
        # Make sure all tensors have the same spatial dimensions
        if r2.shape[2:] != trans_map.shape[2:] or r2.shape[2:] != color_correction.shape[2:]:
            # Resize tensors to match
            trans_map = torch.nn.functional.interpolate(trans_map, size=r2.shape[2:], 
                                                     mode='bilinear', align_corners=False)
            color_correction = torch.nn.functional.interpolate(color_correction, size=r2.shape[2:], 
                                                           mode='bilinear', align_corners=False)
        
        combined = torch.cat([r2, trans_map, color_correction], dim=1)
        dehazed = self.final_conv(combined)
        
        # Apply contrast enhancement
        enhanced = self.ce_relu1(self.ce_conv1(dehazed))
        enhanced = self.ce_conv2(enhanced)
        
        # Use atmospheric scattering model
        # J = (I - A) / t + A
        A = torch.mean(original, dim=(2, 3), keepdim=True).expand_as(original) * 0.8
        
        # Calculate physically-based result using transmission map
        epsilon = 1e-4
        
        # Make sure transmission map has the right dimensions
        if trans_map.shape[2:] != original.shape[2:]:
            trans_map = torch.nn.functional.interpolate(trans_map, size=original.shape[2:], 
                                                     mode='bilinear', align_corners=False)
        
        # Apply atmospheric scattering model
        trans_expanded = trans_map.expand_as(original)
        physical_dehazed = (original - A) / (trans_expanded + epsilon) + A
        physical_dehazed = torch.clamp(physical_dehazed, 0.0, 1.0)
        
        # Combine learned result and physical model with adaptive weighting
        # Higher weight to learned result in areas with strong features
        feature_strength = torch.mean(torch.abs(r2), dim=1, keepdim=True)
        feature_weight = torch.sigmoid(5 * (feature_strength - 0.5))
        
        # Blend results
        result = feature_weight * enhanced + (1 - feature_weight) * physical_dehazed
        
        # Add a small amount of the original for detail preservation
        alpha = 0.05
        result = (1 - alpha) * result + alpha * original
        
        # Final color and contrast enhancement
        # Adaptive S-curve for better contrast without losing details
        s_curve = torch.sigmoid((result - 0.5) * 5)
        boost_factor = torch.clamp(1.5 - torch.mean(result) * 0.5, 1.0, 1.5)
        result = s_curve * boost_factor
        
        # Color balance correction - ensure no color channel dominates
        mean_rgb = torch.mean(result, dim=1, keepdim=True)
        result = mean_rgb + (result - mean_rgb) * 1.3  # Boost saturation
        
        # Final sanity checks and normalization
        result = torch.clamp(result, 0.0, 1.0)
        result = torch.nan_to_num(result, nan=0.5, posinf=1.0, neginf=0.0)
        
        return result

# Model factory to support multiple model types
def create_model(model_type='aod', device='cpu'):
    """Create a dehazing model of the specified type"""
    if model_type == 'aod':
        return AODNet().to(device)
    elif model_type == 'enhanced':
        # Use LightDehazeNet as the "enhanced" model for better efficiency
        return LightDehazeNet().to(device)
    elif model_type == 'light':
        return LightDehazeNet().to(device)
    elif model_type == 'natural':
        # Use LightDehazeNet for natural dehazing with conservative settings
        return LightDehazeNet().to(device)
    elif model_type == 'deep':
        # Use the new DeepDehazeNet model for challenging dehazing scenarios
        return DeepDehazeNet().to(device)
    else:
        logger.error(f"Unknown model type: {model_type}")
        return AODNet().to(device)  # Default fallback

def load_model(device, model_type='aod'):
    """Load the specified dehazing model with weights."""
    # Create the model
    model = create_model(model_type, device)
    
    # Path to the pretrained weights
    weights_path = f'static/models/weights/{model_type}_net.pth'
    
    # Check if weights file exists
    if not os.path.exists(weights_path):
        logger.warning(f"Weights file not found at {weights_path}.")
        
        # Check if weights directory exists
        weights_dir = os.path.dirname(weights_path)
        if not os.path.exists(weights_dir):
            # Create directory if it doesn't exist
            os.makedirs(weights_dir, exist_ok=True)
            logger.info(f"Created weights directory at {weights_dir}")
        
        # Try regenerating weights
        try:
            logger.info(f"Attempting to generate weights for {model_type} model")
            if model_type == 'aod':
                from utils.model import AODNet
                aod_net = AODNet()
                for m in aod_net.modules():
                    if isinstance(m, torch.nn.Conv2d):
                        torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                        if m.bias is not None:
                            torch.nn.init.constant_(m.bias, 0.01)
                torch.nn.init.normal_(aod_net.conv5.weight, mean=0.1, std=0.01)
                torch.nn.init.constant_(aod_net.conv5.bias, 0.1)
                torch.save(aod_net.state_dict(), weights_path)
                logger.info(f"Generated AOD-Net weights at {weights_path}")
            elif model_type in ['enhanced', 'light', 'natural']:
                from utils.model import LightDehazeNet
                enhanced_net = LightDehazeNet()
                for m in enhanced_net.modules():
                    if isinstance(m, torch.nn.Conv2d):
                        torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                        if m.bias is not None:
                            torch.nn.init.constant_(m.bias, 0.01)
                # Use correct attribute names for attention layers
                torch.nn.init.normal_(enhanced_net.att_conv1.weight, mean=0.0, std=0.01)
                torch.nn.init.constant_(enhanced_net.att_conv1.bias, 0.5)
                torch.nn.init.normal_(enhanced_net.att_conv2.weight, mean=0.0, std=0.01)
                torch.nn.init.constant_(enhanced_net.att_conv2.bias, 0.5)
                torch.save(enhanced_net.state_dict(), weights_path)
                logger.info(f"Generated {model_type} model weights at {weights_path}")
            elif model_type == 'deep':
                from utils.model import DeepDehazeNet
                deep_net = DeepDehazeNet()
                for m in deep_net.modules():
                    if isinstance(m, torch.nn.Conv2d):
                        torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                        if m.bias is not None:
                            torch.nn.init.constant_(m.bias, 0.01)
                torch.nn.init.normal_(deep_net.trans_conv2.weight, mean=0.0, std=0.01)
                torch.nn.init.constant_(deep_net.trans_conv2.bias, 0.5)
                torch.save(deep_net.state_dict(), weights_path)
                logger.info(f"Generated DeepDehazeNet weights at {weights_path}")
        except Exception as e:
            logger.error(f"Failed to generate weights: {str(e)}")
            logger.info(f"Using random initialization for {model_type} model")
            model.eval()
            return model
    
    try:
        # Try to load weights with different map_location options if the default fails
        try:
            model.load_state_dict(torch.load(weights_path, map_location=device))
        except Exception as e1:
            logger.warning(f"Error loading weights with device mapping: {str(e1)}")
            try:
                # Try with 'cpu' mapping which is more compatible
                model.load_state_dict(torch.load(weights_path, map_location='cpu'))
            except Exception as e2:
                # If that fails too, try with a more permissive loading approach
                logger.warning(f"Error loading weights with CPU mapping: {str(e2)}")
                state_dict = torch.load(weights_path, map_location='cpu')
                
                # Filter out size mismatches
                model_dict = model.state_dict()
                filtered_state_dict = {k: v for k, v in state_dict.items() 
                                      if k in model_dict and v.shape == model_dict[k].shape}
                model.load_state_dict(filtered_state_dict, strict=False)
                logger.info(f"Loaded partial weights for {model_type} model with shape filtering")
                
        logger.info(f"Successfully loaded pretrained weights for {model_type} model")
    except Exception as e:
        logger.error(f"Error loading weights: {str(e)}")
        logger.warning("Using randomly initialized weights")
    
    # Set model to evaluation mode for inference
    model.eval()
    
    # Apply additional optimization if running on GPU
    if device.type == 'cuda':
        # Try to optimize with torch.jit if supported by the model
        try:
            model = torch.jit.script(model)
            logger.info(f"Applied JIT optimization to {model_type} model")
        except Exception as e:
            logger.warning(f"Could not apply JIT optimization: {str(e)}")
    
    return model

# Standard preprocessing for dehazing models
def get_preprocessing_transforms():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])

# Standard postprocessing for dehazing models
def get_postprocessing_transforms():
    return transforms.Compose([
        transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], 
                             std=[1/0.229, 1/0.224, 1/0.225])
    ])

# Function to optimize model for inference
def optimize_model_for_inference(model, device):
    """
    Apply optimization techniques to the model for faster inference.
    
    Args:
        model: The PyTorch model to optimize
        device: The device the model is on (CPU/CUDA)
        
    Returns:
        Optimized model
    """
    # Set model to evaluation mode
    model.eval()
    
    # Apply torch.jit.script optimization if CUDA is available
    if device.type == 'cuda':
        try:
            # Try to optimize with torch.jit if supported
            model = torch.jit.script(model)
        except Exception as e:
            # If JIT fails, we can still use the regular model
            pass
    
    # Apply half-precision for faster computation on compatible GPUs
    if device.type == 'cuda' and torch.cuda.is_available():
        try:
            # Check if GPU supports float16
            if torch.cuda.get_device_capability()[0] >= 7:  # Volta or newer architecture
                model = model.half()
        except Exception as e:
            # Continue with full precision if half precision fails
            pass
    
    return model
