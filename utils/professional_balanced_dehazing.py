"""
Professional Balanced Dehazing Utility
=====================================

This module provides a professional dehazing system that achieves:
- Crystal clear visibility without aggressive artifacts
- Natural color preservation without purple/blue tints
- Clean, neat results without blending issues
- Perfect balance between clarity and naturalness
"""

import os
import logging
import torch
import torch.nn as nn
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional
import time

logger = logging.getLogger(__name__)

class ProfessionalBalancedNet(nn.Module):
    """Professional Balanced Dehazing Network"""
    
    def __init__(self):
        super(ProfessionalBalancedNet, self).__init__()
        
        # Enhanced encoder with residual connections
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)

        # Professional attention mechanism
        self.attention = nn.Sequential(
            nn.Conv2d(512, 256, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, 1),
            nn.Sigmoid()
        )

        # Enhanced decoder with skip connections
        self.upconv4 = nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1)
        self.upbn4 = nn.BatchNorm2d(256)
        self.upconv3 = nn.ConvTranspose2d(512, 128, 4, stride=2, padding=1)
        self.upbn3 = nn.BatchNorm2d(128)
        self.upconv2 = nn.ConvTranspose2d(256, 64, 4, stride=2, padding=1)
        self.upbn2 = nn.BatchNorm2d(64)
        self.upconv1 = nn.ConvTranspose2d(128, 32, 4, stride=2, padding=1)
        self.upbn1 = nn.BatchNorm2d(32)
        
        # Final output layer
        self.final_conv = nn.Conv2d(32, 3, 3, padding=1)

        # Activations
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Store input for residual connection
        input_img = x
        
        # Encoder with residual connections and downsampling
        e1 = self.relu(self.bn1(self.conv1(x)))
        e1_down = nn.functional.max_pool2d(e1, 2)
        
        e2 = self.relu(self.bn2(self.conv2(e1_down)))
        e2_down = nn.functional.max_pool2d(e2, 2)
        
        e3 = self.relu(self.bn3(self.conv3(e2_down)))
        e3_down = nn.functional.max_pool2d(e3, 2)
        
        e4 = self.relu(self.bn4(self.conv4(e3_down)))
        e4_down = nn.functional.max_pool2d(e4, 2)

        # Professional attention mechanism
        att = self.attention(e4_down)
        e4_att = e4_down * att

        # Decoder with skip connections (fixed dimensions)
        d4 = self.relu(self.upbn4(self.upconv4(e4_att)))
        # Skip connection: make sure dimensions match
        if d4.shape[2:] != e3.shape[2:]:
            e3_resized = nn.functional.interpolate(e3, size=d4.shape[2:], mode='bilinear', align_corners=False)
        else:
            e3_resized = e3
        d4 = torch.cat([d4, e3_resized], dim=1)
        
        d3 = self.relu(self.upbn3(self.upconv3(d4)))
        # Skip connection: make sure dimensions match
        if d3.shape[2:] != e2.shape[2:]:
            e2_resized = nn.functional.interpolate(e2, size=d3.shape[2:], mode='bilinear', align_corners=False)
        else:
            e2_resized = e2
        d3 = torch.cat([d3, e2_resized], dim=1)
        
        d2 = self.relu(self.upbn2(self.upconv2(d3)))
        # Skip connection: make sure dimensions match
        if d2.shape[2:] != e1.shape[2:]:
            e1_resized = nn.functional.interpolate(e1, size=d2.shape[2:], mode='bilinear', align_corners=False)
        else:
            e1_resized = e1
        d2 = torch.cat([d2, e1_resized], dim=1)
        
        d1 = self.relu(self.upbn1(self.upconv1(d2)))
        
        # Final output
        output = self.final_conv(d1)
        
        # Ensure output matches input size
        if output.shape[2:] != input_img.shape[2:]:
            output = nn.functional.interpolate(output, size=input_img.shape[2:], mode='bilinear', align_corners=False)
        
        # Professional balance: Strong dehazing with natural preservation
        dehazed = self.sigmoid(output)
        balanced_output = dehazed * 0.85 + input_img * 0.15

        return torch.clamp(balanced_output, 0, 1)

class ProfessionalBalancedDehazer:
    """Professional Balanced Dehazing System"""
    
    def __init__(self):
        self.name = "Professional Balanced Dehazer"
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.model_loaded = False
        
        # Professional balanced parameters
        self.params = {
            # Core dehazing - balanced for professional results
            'omega': 0.88,                    # Strong but controlled haze removal
            'min_transmission': 0.12,         # Balanced minimum transmission
            'dark_channel_kernel': 12,        # Professional kernel size
            'guided_filter_radius': 35,       # Smooth but detailed
            'guided_filter_epsilon': 0.001,   # Natural edge preservation
            'atmospheric_percentile': 98.0,   # Conservative atmospheric estimation
            
            # Enhancement - natural quality
            'brightness_factor': 1.15,        # Gentle brightness boost
            'contrast_factor': 1.25,          # Professional contrast
            'saturation_factor': 0.98,        # Slight desaturation for naturalness
            'gamma_correction': 0.95,         # Subtle gamma correction
            
            # Color preservation - prevent tints
            'color_balance_strength': 0.8,    # Strong color balance
            'white_balance_strength': 0.7,    # Natural white balance
            'temperature_adjustment': 0.02,   # Minimal temperature shift
            
            # Final processing - clean results
            'noise_reduction': 0.3,           # Light noise reduction
            'sharpening_strength': 0.6,       # Professional sharpening
            'final_blend_ratio': 0.92,        # Strong but natural blend
            'artifact_suppression': True      # Remove artifacts
        }
        
        # Try to load trained model
        self.load_trained_model()
        
        logger.info("Professional Balanced Dehazer initialized")
    
    def load_trained_model(self):
        """Load the trained professional balanced model"""
        model_paths = [
            "models/professional_balanced_dehazing/professional_balanced_model.pth",
            "models/perfect_balanced_dehazing/perfect_balanced_model.pth",
            "models/improved_perfect_balanced/improved_perfect_model.pth"
        ]
        
        for model_path in model_paths:
            if os.path.exists(model_path):
                try:
                    self.model = ProfessionalBalancedNet().to(self.device)
                    checkpoint = torch.load(model_path, map_location=self.device)
                    
                    if 'model_state_dict' in checkpoint:
                        self.model.load_state_dict(checkpoint['model_state_dict'])
                    else:
                        self.model.load_state_dict(checkpoint)
                    
                    self.model.eval()
                    self.model_loaded = True
                    logger.info(f"Loaded trained model from: {model_path}")
                    return
                except Exception as e:
                    logger.warning(f"Failed to load model from {model_path}: {str(e)}")
        
        logger.info("No trained model found, using algorithmic approach")
    
    def professional_algorithmic_dehaze(self, image: np.ndarray) -> np.ndarray:
        """Professional algorithmic dehazing with perfect balance"""
        
        # Step 1: Dark channel prior
        dark_channel = self.calculate_dark_channel(image, self.params['dark_channel_kernel'])
        
        # Step 2: Atmospheric light estimation (conservative)
        atmospheric_light = self.estimate_atmospheric_light(image, dark_channel, self.params['atmospheric_percentile'])
        
        # Step 3: Transmission estimation
        transmission = self.estimate_transmission(image, atmospheric_light, self.params['omega'])
        transmission = cv2.max(transmission, self.params['min_transmission'])
        
        # Step 4: Guided filter refinement
        transmission_refined = self.guided_filter(
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
            transmission,
            self.params['guided_filter_radius'],
            self.params['guided_filter_epsilon']
        )
        
        # Step 5: Recover scene radiance
        recovered = self.recover_scene_radiance(image, transmission_refined, atmospheric_light)
        
        # Step 6: Professional enhancement
        enhanced = self.professional_enhancement(recovered)
        
        # Step 7: Color balance and artifact removal
        balanced = self.professional_color_balance(enhanced)
        
        # Step 8: Final blend for natural results
        final_result = cv2.addWeighted(
            balanced, self.params['final_blend_ratio'],
            image, 1 - self.params['final_blend_ratio'],
            0
        )
        
        # Step 9: Artifact suppression
        if self.params['artifact_suppression']:
            final_result = self.suppress_artifacts(final_result)
        
        return final_result
    
    def dehaze_with_trained_model(self, image: np.ndarray) -> np.ndarray:
        """Dehaze using trained model"""
        try:
            # Preprocess
            original_size = image.shape[:2]
            image_resized = cv2.resize(image, (256, 256))
            image_norm = image_resized.astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_norm.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
            
            # Inference
            with torch.no_grad():
                output_tensor = self.model(image_tensor)
            
            # Postprocess
            output_np = output_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            output_np = np.clip(output_np, 0, 1)
            dehazed = (output_np * 255).astype(np.uint8)
            
            # Resize back to original size
            dehazed = cv2.resize(dehazed, (original_size[1], original_size[0]))
            
            return dehazed
            
        except Exception as e:
            logger.error(f"Trained model inference failed: {str(e)}")
            return self.professional_algorithmic_dehaze(image)
    
    def calculate_dark_channel(self, image: np.ndarray, kernel_size: int) -> np.ndarray:
        """Calculate dark channel prior"""
        min_channel = np.min(image, axis=2)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        dark_channel = cv2.erode(min_channel, kernel)
        return dark_channel
    
    def estimate_atmospheric_light(self, image: np.ndarray, dark_channel: np.ndarray, percentile: float) -> np.ndarray:
        """Estimate atmospheric light conservatively"""
        num_pixels = int(max(dark_channel.shape[0] * dark_channel.shape[1] * 0.001, 1))
        dark_vec = dark_channel.reshape(-1)
        image_vec = image.reshape(-1, 3)
        
        indices = np.argpartition(dark_vec, -num_pixels)[-num_pixels:]
        brightest_pixels = image_vec[indices]
        
        # Conservative estimate
        atmospheric_light = np.percentile(brightest_pixels, percentile, axis=0)
        
        # Ensure reasonable values
        atmospheric_light = np.clip(atmospheric_light, 180, 255)
        
        return atmospheric_light
    
    def estimate_transmission(self, image: np.ndarray, atmospheric_light: np.ndarray, omega: float) -> np.ndarray:
        """Estimate transmission map"""
        normalized_image = image.astype(np.float64) / atmospheric_light
        transmission = 1 - omega * np.min(normalized_image, axis=2)
        return transmission
    
    def guided_filter(self, guide: np.ndarray, src: np.ndarray, radius: int, epsilon: float) -> np.ndarray:
        """Guided filter for edge-preserving smoothing"""
        guide = guide.astype(np.float64) / 255.0
        src = src.astype(np.float64)
        
        mean_guide = cv2.boxFilter(guide, cv2.CV_64F, (radius, radius))
        mean_src = cv2.boxFilter(src, cv2.CV_64F, (radius, radius))
        mean_guide_src = cv2.boxFilter(guide * src, cv2.CV_64F, (radius, radius))
        
        cov_guide_src = mean_guide_src - mean_guide * mean_src
        mean_guide_sq = cv2.boxFilter(guide * guide, cv2.CV_64F, (radius, radius))
        var_guide = mean_guide_sq - mean_guide * mean_guide
        
        a = cov_guide_src / (var_guide + epsilon)
        b = mean_src - a * mean_guide
        
        mean_a = cv2.boxFilter(a, cv2.CV_64F, (radius, radius))
        mean_b = cv2.boxFilter(b, cv2.CV_64F, (radius, radius))
        
        output = mean_a * guide + mean_b
        
        return output
    
    def recover_scene_radiance(self, image: np.ndarray, transmission: np.ndarray, atmospheric_light: np.ndarray) -> np.ndarray:
        """Recover scene radiance"""
        image = image.astype(np.float64)
        transmission = transmission[:, :, np.newaxis]
        
        recovered = np.zeros_like(image)
        for c in range(3):
            recovered[:, :, c] = (image[:, :, c] - atmospheric_light[c]) / transmission[:, :, 0] + atmospheric_light[c]
        
        return np.clip(recovered, 0, 255).astype(np.uint8)
    
    def professional_enhancement(self, image: np.ndarray) -> np.ndarray:
        """Professional quality enhancement"""
        enhanced = image.copy().astype(np.float32)
        
        # Brightness adjustment
        enhanced = enhanced * self.params['brightness_factor']
        
        # Contrast adjustment
        enhanced = ((enhanced / 255.0 - 0.5) * self.params['contrast_factor'] + 0.5) * 255.0
        
        # Gamma correction
        enhanced = np.power(enhanced / 255.0, self.params['gamma_correction']) * 255.0
        
        # Saturation adjustment
        hsv = cv2.cvtColor(enhanced.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = hsv[:, :, 1] * self.params['saturation_factor']
        enhanced = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        return np.clip(enhanced, 0, 255).astype(np.uint8)
    
    def professional_color_balance(self, image: np.ndarray) -> np.ndarray:
        """Professional color balance to prevent tints"""
        balanced = image.copy().astype(np.float32)
        
        # Calculate channel means
        b_mean, g_mean, r_mean = cv2.mean(image)[:3]
        total_mean = (b_mean + g_mean + r_mean) / 3
        
        # Calculate balance factors
        b_factor = total_mean / b_mean if b_mean > 0 else 1.0
        g_factor = total_mean / g_mean if g_mean > 0 else 1.0
        r_factor = total_mean / r_mean if r_mean > 0 else 1.0
        
        # Apply controlled balance
        strength = self.params['color_balance_strength']
        b_factor = 1.0 + (b_factor - 1.0) * strength
        g_factor = 1.0 + (g_factor - 1.0) * strength
        r_factor = 1.0 + (r_factor - 1.0) * strength
        
        # Limit factors to prevent overcorrection
        b_factor = np.clip(b_factor, 0.85, 1.15)
        g_factor = np.clip(g_factor, 0.85, 1.15)
        r_factor = np.clip(r_factor, 0.85, 1.15)
        
        # Apply balance
        balanced[:, :, 0] *= b_factor
        balanced[:, :, 1] *= g_factor
        balanced[:, :, 2] *= r_factor
        
        return np.clip(balanced, 0, 255).astype(np.uint8)
    
    def suppress_artifacts(self, image: np.ndarray) -> np.ndarray:
        """Suppress artifacts and noise"""
        # Light bilateral filtering to remove artifacts
        filtered = cv2.bilateralFilter(image, 5, 30, 30)
        
        # Professional sharpening
        if self.params['sharpening_strength'] > 0:
            gaussian = cv2.GaussianBlur(filtered, (0, 0), 1.0)
            sharpened = cv2.addWeighted(filtered, 1.0 + self.params['sharpening_strength'], gaussian, -self.params['sharpening_strength'], 0)
            filtered = sharpened
        
        # Noise reduction
        if self.params['noise_reduction'] > 0:
            denoised = cv2.fastNlMeansDenoisingColored(filtered, None, self.params['noise_reduction'] * 10, self.params['noise_reduction'] * 10, 7, 21)
            filtered = denoised
        
        return filtered
    
    def dehaze_image(self, image: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Main dehazing function"""
        start_time = time.time()
        
        try:
            # Use trained model if available, otherwise algorithmic approach
            if self.model_loaded:
                result = self.dehaze_with_trained_model(image)
                method_used = "Professional Trained Model"
            else:
                result = self.professional_algorithmic_dehaze(image)
                method_used = "Professional Algorithmic"
            
            processing_time = time.time() - start_time
            
            # Calculate quality metrics
            metrics = self.calculate_quality_metrics(image, result)
            
            info = {
                'method': method_used,
                'processing_time': processing_time,
                'model_loaded': self.model_loaded,
                'quality_metrics': metrics,
                'parameters_used': self.params
            }
            
            logger.info(f"Professional dehazing completed in {processing_time:.3f}s using {method_used}")
            
            return result, info
            
        except Exception as e:
            logger.error(f"Professional dehazing failed: {str(e)}")
            raise
    
    def calculate_quality_metrics(self, original: np.ndarray, processed: np.ndarray) -> Dict:
        """Calculate comprehensive quality metrics"""
        try:
            # Brightness and contrast
            orig_mean = np.mean(original)
            proc_mean = np.mean(processed)
            brightness_improvement = (proc_mean - orig_mean) / 255.0
            
            orig_std = np.std(original)
            proc_std = np.std(processed)
            contrast_improvement = (proc_std - orig_std) / 255.0
            
            # Color balance
            orig_channels = cv2.mean(original)[:3]
            proc_channels = cv2.mean(processed)[:3]
            
            orig_balance = np.std(orig_channels)
            proc_balance = np.std(proc_channels)
            color_balance_improvement = (orig_balance - proc_balance) / 255.0
            
            # Edge density (clarity)
            orig_edges = cv2.Canny(cv2.cvtColor(original, cv2.COLOR_BGR2GRAY), 50, 150)
            proc_edges = cv2.Canny(cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY), 50, 150)
            
            orig_edge_density = np.sum(orig_edges > 0) / orig_edges.size
            proc_edge_density = np.sum(proc_edges > 0) / proc_edges.size
            clarity_improvement = proc_edge_density - orig_edge_density
            
            return {
                'brightness_improvement': brightness_improvement,
                'contrast_improvement': contrast_improvement,
                'color_balance_improvement': color_balance_improvement,
                'clarity_improvement': clarity_improvement,
                'overall_quality': (brightness_improvement + contrast_improvement + clarity_improvement) / 3
            }
        except:
            return {
                'brightness_improvement': 0.0,
                'contrast_improvement': 0.0,
                'color_balance_improvement': 0.0,
                'clarity_improvement': 0.0,
                'overall_quality': 0.0
            }

def professional_balanced_dehaze(input_path: str, output_dir: str, device: str = 'cpu') -> str:
    """
    Professional Balanced Dehazing Function
    
    Args:
        input_path: Path to input hazy image
        output_dir: Directory to save output
        device: Device to use ('cpu' or 'cuda')
    
    Returns:
        Path to output image
    """
    try:
        # Initialize dehazer
        dehazer = ProfessionalBalancedDehazer()
        
        # Load image
        image = cv2.imread(input_path)
        if image is None:
            raise ValueError(f"Could not load image: {input_path}")
        
        # Dehaze image
        result, info = dehazer.dehaze_image(image)
        
        # Save result
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}_professional_balanced.jpg")
        
        cv2.imwrite(output_path, result)
        logger.info(f"Professional balanced dehazing completed: {output_path}")
        logger.info(f"Quality improvement: {info['quality_metrics']['overall_quality']:.3f}")
        
        return output_path
        
    except Exception as e:
        logger.error(f"Professional balanced dehazing failed: {str(e)}")
        raise
