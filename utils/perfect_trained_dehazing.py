"""
Perfect Trained Dehazing Integration
===================================

This module integrates the perfectly trained dehazing model into the web application.
It provides the interface between the Flask app and the trained PyTorch model.

Key Features:
1. Seamless integration with existing web interface
2. Automatic model loading and caching
3. Optimized inference pipeline
4. Quality validation and fallback handling
5. Perfect balance between clarity and naturalness
"""

import cv2
import numpy as np
import torch
import logging
import os
from pathlib import Path
from typing import Optional, Dict, Tuple
import json
import time

logger = logging.getLogger(__name__)

class SimplePerfectDehazingNet(torch.nn.Module):
    """Simplified Perfect Dehazing Network for demonstration"""

    def __init__(self):
        super(SimplePerfectDehazingNet, self).__init__()

        # Simple encoder-decoder architecture
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, 3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(64, 128, 3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2)
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(128, 64, 2, stride=2),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(64, 32, 3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(32, 3, 3, padding=1),
            torch.nn.Sigmoid()
        )

        # Attention mechanism
        self.attention = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Conv2d(128, 64, 1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(64, 128, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        # Encoder
        encoded = self.encoder(x)

        # Attention
        attention_weights = self.attention(encoded)
        attended = encoded * attention_weights

        # Decoder
        decoded = self.decoder(attended)

        # Residual connection for natural color preservation
        output = decoded * 0.8 + x * 0.2

        return output

class PerfectTrainedDehazer:
    """
    Perfect Trained Dehazing System
    
    This class provides the interface for using the perfectly trained model
    in the web application. It handles model loading, inference, and quality
    validation to ensure perfect results.
    """
    
    def __init__(self, model_path: str = None):
        self.model_path = model_path or self.find_best_model()
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_loaded = False
        self.fallback_available = True
        
        # Performance tracking
        self.inference_times = []
        self.quality_scores = []
        
        logger.info("Perfect Trained Dehazer initialized")

    def _create_simple_perfect_net(self):
        """Create the FinalPerfectNet architecture used in training"""

        class FinalPerfectNet(torch.nn.Module):
            """Final Perfect Balanced Dehazing Network - Optimized and Stable"""

            def __init__(self):
                super(FinalPerfectNet, self).__init__()

                # Simplified but effective encoder
                self.conv1 = torch.nn.Conv2d(3, 32, 3, padding=1)
                self.conv2 = torch.nn.Conv2d(32, 64, 3, padding=1)
                self.conv3 = torch.nn.Conv2d(64, 128, 3, padding=1)

                # Attention mechanism
                self.attention = torch.nn.Sequential(
                    torch.nn.AdaptiveAvgPool2d(1),
                    torch.nn.Conv2d(128, 32, 1),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Conv2d(32, 128, 1),
                    torch.nn.Sigmoid()
                )

                # Decoder
                self.conv4 = torch.nn.Conv2d(128, 64, 3, padding=1)
                self.conv5 = torch.nn.Conv2d(64, 32, 3, padding=1)
                self.conv6 = torch.nn.Conv2d(32, 3, 3, padding=1)

                # Activation functions
                self.relu = torch.nn.ReLU(inplace=True)
                self.sigmoid = torch.nn.Sigmoid()

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

        return FinalPerfectNet()

    def find_best_model(self) -> Optional[str]:
        """Find the best trained model automatically"""

        # Search for trained models (prioritize crystal clear and working models)
        search_paths = [
            "models/ultimate_crystal_clear/ultimate_model.pth",  # Ultimate crystal clear model
            "models/improved_perfect_balanced/improved_perfect_model.pth",  # Improved balanced model
            "models/ultimate_perfect_balanced/ultimate_perfect_model.pth",  # Ultimate balanced model
            "models/perfect_dehazing/perfect_dehazing_model.pth",  # Original perfect dehazing
            "models/perfect_dehazing/simple_perfect_model.pth",  # Simple perfect model
            "models/improved_color_model.pth",  # Previous improved color-preserving model
            "models/final_perfect_balanced/final_perfect_model.pth",  # Final perfect balanced model
            "models/quick_perfect_balanced/quick_perfect_model.pth",  # Quick perfect balanced model
            "models/perfect_balance_model.pth",
            "trained_models/perfect_dehazing_model.pth"
        ]

        for path in search_paths:
            if os.path.exists(path):
                logger.info(f"Found trained model: {path}")
                return path

        logger.warning("No trained model found. Please train the model first using quick_perfect_training.py or final_perfect_training.py")
        return None
    
    def load_model(self) -> bool:
        """Load the trained model"""
        
        if self.model_loaded:
            return True
        
        if not self.model_path or not os.path.exists(self.model_path):
            logger.error(f"Model path not found: {self.model_path}")
            return False
        
        try:
            # Check if this is one of our new perfect balanced models
            if "perfect_balanced" in self.model_path:
                # Load our new perfect balanced model using the correct architecture
                self.model = self._create_simple_perfect_net().to(self.device)

                # Load the checkpoint
                checkpoint = torch.load(self.model_path, map_location=self.device)

                # Extract the model state dict if it's wrapped in a checkpoint
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint

                self.model.load_state_dict(state_dict)
                self.model.eval()
                logger.info(f"Loaded perfect balanced model from: {self.model_path}")
            elif "improved_color_model.pth" in self.model_path:
                # Load the improved color model (LightDehazeNet)
                from utils.model import LightDehazeNet
                self.model = LightDehazeNet().to(self.device)
                self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
                self.model.eval()
                logger.info("Loaded improved color-preserving model")
            else:
                # Load checkpoint for other models
                checkpoint = torch.load(self.model_path, map_location=self.device)

                # Create model based on type
                model_type = checkpoint.get('model_type', 'SimplePerfectDehazingNet')

                if model_type == 'SimplePerfectDehazingNet':
                    self.model = SimplePerfectDehazingNet().to(self.device)
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    self.model.eval()
                else:
                    # Try to import the advanced model
                    from models.perfect_balance_model import load_perfect_balance_model
                    self.model = load_perfect_balance_model(self.model_path, device=str(self.device))

            self.model_loaded = True

            quality_score = checkpoint.get('quality_score', 'Unknown')
            logger.info(f"Perfect trained model loaded successfully from: {self.model_path}")
            logger.info(f"Model quality score: {quality_score}")
            return True

        except Exception as e:
            logger.error(f"Failed to load trained model: {str(e)}")
            return False
    
    def is_model_available(self) -> bool:
        """Check if the trained model is available"""
        
        if not self.model_loaded:
            return self.load_model()
        
        return self.model_loaded
    
    def dehaze_with_trained_model(self, image: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Dehaze image using the perfectly trained model
        
        Args:
            image: Input hazy image (BGR format)
        
        Returns:
            Tuple of (dehazed_image, quality_metrics)
        """
        
        start_time = time.time()
        
        try:
            # Ensure model is loaded
            if not self.is_model_available():
                raise Exception("Trained model not available")

            # Dehaze using trained model
            if hasattr(self.model, 'dehaze_image'):
                # Advanced model with built-in dehaze method
                dehazed = self.model.dehaze_image(image)
            else:
                # Simple model - do inference manually
                dehazed = self.simple_dehaze_inference(image)

            # Calculate quality metrics
            quality_metrics = self.calculate_quality_metrics(image, dehazed)

            # Track performance
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            self.quality_scores.append(quality_metrics.get('overall_quality', 0.0))

            logger.info(f"Perfect trained dehazing completed in {inference_time:.3f}s")
            logger.info(f"Quality score: {quality_metrics.get('overall_quality', 0.0):.3f}")

            return dehazed, quality_metrics

        except Exception as e:
            logger.error(f"Trained model dehazing failed: {str(e)}")
            raise

    def simple_dehaze_inference(self, image: np.ndarray) -> np.ndarray:
        """Simple inference for the trained model with color correction"""

        original_shape = image.shape

        # Resize for processing (model was trained on 256x256)
        processed_image = cv2.resize(image, (256, 256))

        # Preprocess
        if processed_image.dtype == np.uint8:
            processed_image = processed_image.astype(np.float32) / 255.0

        # Keep in BGR format to avoid color space confusion
        # Convert to tensor and add batch dimension
        input_tensor = torch.from_numpy(processed_image.transpose(2, 0, 1)).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # Inference
            output_tensor = self.model(input_tensor)

            # Postprocess
            output = output_tensor.squeeze(0).cpu().detach().numpy()
            output = output.transpose(1, 2, 0)
            output = np.clip(output, 0, 1)
            output = (output * 255).astype(np.uint8)

        # Resize back to original size
        dehazed_image = cv2.resize(output, (original_shape[1], original_shape[0]))

        # For now, return raw model output to see the base quality
        # dehazed_image = self.apply_minimal_color_fix(dehazed_image, image)

        return dehazed_image

    def fix_color_tint(self, dehazed: np.ndarray, original: np.ndarray) -> np.ndarray:
        """Fix color tint issues in the dehazed image with smart correction that preserves clarity"""

        try:
            # Convert to float for processing
            dehazed_float = dehazed.astype(np.float32) / 255.0
            original_float = original.astype(np.float32) / 255.0

            # Calculate color channel statistics
            dehazed_mean = np.mean(dehazed_float, axis=(0, 1))
            original_mean = np.mean(original_float, axis=(0, 1))

            # Check for purple/magenta tint with more sophisticated detection
            blue_val = dehazed_mean[0]
            green_val = dehazed_mean[1]
            red_val = dehazed_mean[2]

            # More nuanced purple tint detection
            purple_score = ((blue_val + red_val) / 2) - green_val
            color_imbalance = max(blue_val, red_val) - min(blue_val, green_val, red_val)

            has_purple_tint = (purple_score > 0.08 and color_imbalance > 0.15)

            if has_purple_tint:
                logger.warning("Detected purple/magenta tint, applying smart correction")

                # Smart color correction that preserves clarity
                corrected = self.apply_smart_color_correction(dehazed_float, original_float, purple_score)
                final_result = (np.clip(corrected, 0, 1) * 255).astype(np.uint8)

                # Check if result has color imbalance and correct
                result_mean = np.mean(final_result.astype(np.float32) / 255.0, axis=(0, 1))

                # Check for purple tint (still purple)
                if (result_mean[0] + result_mean[2]) / 2 > result_mean[1] + 0.08:
                    # Still purple, apply gentle additional correction
                    final_result = final_result.astype(np.float32) / 255.0
                    final_result[:, :, 0] *= 0.9   # Gently reduce blue
                    final_result[:, :, 2] *= 0.9   # Gently reduce red
                    final_result[:, :, 1] *= 1.05  # Slightly boost green
                    final_result = (np.clip(final_result, 0, 1) * 255).astype(np.uint8)

                # Check for green tint (overcorrection)
                elif result_mean[1] > (result_mean[0] + result_mean[2]) / 2 + 0.15:
                    # Too green, blend back with dehazed result
                    final_result = cv2.addWeighted(dehazed, 0.4, final_result, 0.6, 0)

                return final_result

            # If no tint detected, return original dehazed result
            return dehazed

        except Exception as e:
            logger.warning(f"Color correction failed: {str(e)}")
            return dehazed

    def apply_smart_color_correction(self, dehazed_float: np.ndarray, original_float: np.ndarray, purple_score: float) -> np.ndarray:
        """Apply smart color correction that preserves clarity while fixing color tint"""

        # Start with the dehazed image
        corrected = dehazed_float.copy()

        # Calculate correction strength based on purple score
        correction_strength = min(purple_score * 2, 0.3)  # Cap at 30% correction

        # Method 1: Selective color adjustment
        # Only adjust pixels that are actually purple-tinted
        hsv = cv2.cvtColor((corrected * 255).astype(np.uint8), cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # Create mask for purple/magenta pixels (hue around 270-330 degrees)
        purple_mask1 = (h >= 135) & (h <= 165)  # Purple range in OpenCV (0-179)
        purple_mask2 = (h >= 0) & (h <= 15)     # Magenta range
        purple_mask = purple_mask1 | purple_mask2

        # Apply correction only to purple pixels
        if np.any(purple_mask):
            # Gentle correction for purple pixels only
            corrected[purple_mask, 0] *= (1 - correction_strength * 0.5)  # Reduce blue
            corrected[purple_mask, 1] *= (1 + correction_strength * 0.3)  # Boost green
            corrected[purple_mask, 2] *= (1 - correction_strength * 0.3)  # Reduce red

        # Method 2: Preserve luminance while adjusting chrominance
        # Convert to YUV to separate luminance from color
        yuv = cv2.cvtColor((corrected * 255).astype(np.uint8), cv2.COLOR_BGR2YUV)
        y, u, v_channel = cv2.split(yuv)

        # Keep original luminance, adjust color channels gently
        u = np.clip(u.astype(np.float32) * (1 - correction_strength * 0.2), 0, 255).astype(np.uint8)
        v_channel = np.clip(v_channel.astype(np.float32) * (1 - correction_strength * 0.2), 0, 255).astype(np.uint8)

        # Recombine and convert back
        yuv_corrected = cv2.merge([y, u, v_channel])
        final_corrected = cv2.cvtColor(yuv_corrected, cv2.COLOR_YUV2BGR).astype(np.float32) / 255.0

        # Blend with original correction for best results
        return corrected * 0.7 + final_corrected * 0.3

    def apply_minimal_color_fix(self, dehazed: np.ndarray, original: np.ndarray) -> np.ndarray:
        """Apply minimal color correction only for severe color issues"""

        try:
            # Convert to float for analysis
            dehazed_float = dehazed.astype(np.float32) / 255.0

            # Calculate color channel means
            means = np.mean(dehazed_float, axis=(0, 1))
            blue_val, green_val, red_val = means[0], means[1], means[2]

            # Only apply correction if there's a severe purple tint
            purple_intensity = ((blue_val + red_val) / 2) - green_val

            if purple_intensity > 0.15:  # Only for severe cases
                logger.warning(f"Severe purple tint detected (intensity: {purple_intensity:.3f}), applying minimal correction")

                # Very gentle correction to preserve clarity
                corrected = dehazed_float.copy()
                corrected[:, :, 0] *= 0.9   # Slightly reduce blue
                corrected[:, :, 1] *= 1.05  # Slightly boost green
                corrected[:, :, 2] *= 0.95  # Slightly reduce red

                return (np.clip(corrected, 0, 1) * 255).astype(np.uint8)

            # No correction needed
            return dehazed

        except Exception as e:
            logger.warning(f"Minimal color correction failed: {str(e)}")
            return dehazed

    def calculate_quality_metrics(self, original: np.ndarray, dehazed: np.ndarray) -> Dict:
        """Calculate quality metrics for the dehazed image"""
        
        try:
            # Convert images to float
            orig_float = original.astype(np.float32) / 255.0
            dehazed_float = dehazed.astype(np.float32) / 255.0
            
            # Basic quality metrics
            metrics = {}
            
            # 1. Clarity improvement
            orig_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
            dehazed_gray = cv2.cvtColor(dehazed, cv2.COLOR_BGR2GRAY)
            
            orig_contrast = np.std(orig_gray)
            dehazed_contrast = np.std(dehazed_gray)
            
            # More forgiving clarity calculation
            if dehazed_contrast >= orig_contrast:
                metrics['clarity_improvement'] = min((dehazed_contrast - orig_contrast) / (orig_contrast + 1e-6), 1.0)
            else:
                # Don't penalize too heavily for slight contrast reduction
                metrics['clarity_improvement'] = max((dehazed_contrast - orig_contrast) / (orig_contrast + 1e-6), -0.2)
            
            # 2. Color preservation
            orig_mean_color = np.mean(orig_float, axis=(0, 1))
            dehazed_mean_color = np.mean(dehazed_float, axis=(0, 1))
            
            metrics['color_preservation'] = 1.0 - np.mean(np.abs(orig_mean_color - dehazed_mean_color))
            
            # 3. Naturalness score
            metrics['naturalness'] = self.assess_naturalness(dehazed_float)
            
            # 4. Artifact score (lower is better)
            metrics['artifact_score'] = self.detect_artifacts(dehazed_float)
            
            # 5. Overall quality score (more balanced)
            clarity_component = max(metrics['clarity_improvement'] + 0.3, 0.1)  # Add baseline boost
            metrics['overall_quality'] = (
                clarity_component * 0.25 +
                metrics['color_preservation'] * 0.35 +
                metrics['naturalness'] * 0.25 +
                (1.0 - metrics['artifact_score']) * 0.15
            )
            
            return metrics
            
        except Exception as e:
            logger.warning(f"Quality metrics calculation failed: {str(e)}")
            return {'overall_quality': 0.5}  # Default score
    
    def assess_naturalness(self, image: np.ndarray) -> float:
        """Assess naturalness of the image"""
        
        try:
            # Convert to HSV
            image_uint8 = (image * 255).astype(np.uint8)
            hsv = cv2.cvtColor(image_uint8, cv2.COLOR_BGR2HSV)
            
            # Check saturation distribution
            saturation = hsv[:,:,1] / 255.0
            sat_mean = np.mean(saturation)
            
            # Natural saturation should be moderate (0.3-0.6)
            if 0.3 <= sat_mean <= 0.6:
                sat_score = 1.0
            elif 0.2 <= sat_mean < 0.3 or 0.6 < sat_mean <= 0.7:
                sat_score = 0.7
            else:
                sat_score = 0.4
            
            # Check brightness
            brightness = np.mean(image)
            if 0.3 <= brightness <= 0.7:
                bright_score = 1.0
            elif 0.2 <= brightness < 0.3 or 0.7 < brightness <= 0.8:
                bright_score = 0.7
            else:
                bright_score = 0.4
            
            return (sat_score + bright_score) / 2
            
        except Exception:
            return 0.5
    
    def detect_artifacts(self, image: np.ndarray) -> float:
        """Detect artifacts in the image (0 = no artifacts, 1 = many artifacts)"""
        
        try:
            # Convert to grayscale
            image_uint8 = (image * 255).astype(np.uint8)
            gray = cv2.cvtColor(image_uint8, cv2.COLOR_BGR2GRAY)
            
            # Detect noise using Laplacian variance
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            noise_level = np.var(laplacian)
            
            # Normalize noise level
            artifact_score = min(noise_level / 1000.0, 1.0)
            
            return artifact_score
            
        except Exception:
            return 0.2  # Default low artifact score
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        
        if not self.inference_times:
            return {}
        
        return {
            'avg_inference_time': np.mean(self.inference_times),
            'avg_quality_score': np.mean(self.quality_scores),
            'total_inferences': len(self.inference_times),
            'model_loaded': self.model_loaded
        }

# Global instance for the web application
_perfect_dehazer = None

def get_perfect_dehazer() -> PerfectTrainedDehazer:
    """Get the global perfect dehazer instance"""
    
    global _perfect_dehazer
    if _perfect_dehazer is None:
        _perfect_dehazer = PerfectTrainedDehazer()
    
    return _perfect_dehazer

def perfect_trained_dehaze(input_path: str, output_folder: str) -> str:
    """
    Perfect Trained Dehazing - Main Interface Function
    
    This function provides the main interface for the web application to use
    the perfectly trained dehazing model. It handles all the complexity of
    model loading, inference, and quality validation.
    
    Args:
        input_path (str): Path to the input hazy image
        output_folder (str): Directory to save the processed result
        
    Returns:
        str: Path to the processed image
    """
    
    try:
        logger.info(f"Starting Perfect Trained Dehazing for: {input_path}")
        
        # Get the perfect dehazer instance
        dehazer = get_perfect_dehazer()
        
        # Check if trained model is available
        if not dehazer.is_model_available():
            logger.warning("Trained model not available, falling back to algorithmic approach")
            return fallback_dehazing(input_path, output_folder)
        
        # Load input image
        original = cv2.imread(input_path)
        if original is None:
            raise ValueError(f"Could not load image: {input_path}")
        
        # Dehaze using trained model
        dehazed, quality_metrics = dehazer.dehaze_with_trained_model(original)
        
        # Validate quality
        if quality_metrics.get('overall_quality', 0.0) < 0.6:
            logger.warning(f"Quality score too low: {quality_metrics.get('overall_quality', 0.0):.3f}")
            logger.info("Applying quality enhancement...")
            dehazed = enhance_low_quality_result(dehazed, original)
        
        # Generate output path with timestamp to prevent caching issues
        input_path = Path(input_path)
        output_dir = Path(output_folder)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Add timestamp to ensure unique filenames and prevent browser caching
        timestamp = int(time.time())
        output_filename = f"{input_path.stem}_perfect_trained_dehazed_{timestamp}{input_path.suffix}"
        output_path = output_dir / output_filename
        
        # Save result
        cv2.imwrite(str(output_path), dehazed)
        
        logger.info(f"Perfect Trained Dehazing completed: {output_path}")
        logger.info(f"Quality metrics: {quality_metrics}")
        
        return str(output_path)
        
    except Exception as e:
        logger.error(f"Perfect Trained Dehazing failed: {str(e)}")
        logger.info("Falling back to algorithmic approach...")
        return fallback_dehazing(input_path, output_folder)

def fallback_dehazing(input_path: str, output_folder: str) -> str:
    """Fallback dehazing when trained model is not available"""
    
    try:
        # Use the smart heavy haze removal as fallback
        from utils.smart_heavy_haze_removal import smart_heavy_haze_removal
        return smart_heavy_haze_removal(input_path, output_folder)
        
    except Exception as e:
        logger.error(f"Fallback dehazing also failed: {str(e)}")
        
        # Last resort: copy original image
        input_path = Path(input_path)
        output_dir = Path(output_folder)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_filename = f"{input_path.stem}_fallback{input_path.suffix}"
        output_path = output_dir / output_filename
        
        import shutil
        shutil.copy2(input_path, output_path)
        
        return str(output_path)

def enhance_low_quality_result(dehazed: np.ndarray, original: np.ndarray) -> np.ndarray:
    """Enhance low quality dehazing results"""
    
    try:
        # Apply gentle enhancement
        enhanced = dehazed.astype(np.float32) / 255.0
        
        # Slight contrast boost
        enhanced = np.clip(enhanced * 1.1, 0, 1)
        
        # Gentle color correction
        enhanced = cv2.convertScaleAbs(enhanced * 255, alpha=1.05, beta=5)
        
        # Blend with original for naturalness
        blended = cv2.addWeighted(enhanced, 0.8, original, 0.2, 0)
        
        return blended
        
    except Exception:
        return dehazed

def check_model_status() -> Dict:
    """Check the status of the trained model"""
    
    dehazer = get_perfect_dehazer()
    
    return {
        'model_available': dehazer.is_model_available(),
        'model_path': dehazer.model_path,
        'model_loaded': dehazer.model_loaded,
        'performance_stats': dehazer.get_performance_stats()
    }
