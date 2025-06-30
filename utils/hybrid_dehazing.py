"""
Advanced Hybrid Dehazing System - Multi-Model Ensemble Approach
Combines multiple models intelligently to achieve results closest to ground truth

This module implements:
1. Multi-model ensemble processing
2. Quality-based model selection
3. Adaptive parameter tuning
4. Ground truth approximation
"""

import cv2
import numpy as np
import torch
import logging
import os
import time
from typing import Dict, List, Tuple, Optional
from scipy.ndimage import gaussian_filter
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
from .dehazing import process_image, dehaze_with_clahe
from .direct_dehazing import natural_dehaze, adaptive_natural_dehaze, conservative_color_dehaze
from .ultra_aggressive_dehazing import ultra_aggressive_dehaze

logger = logging.getLogger(__name__)

class AdvancedDehazingEnsemble:
    """
    Advanced ensemble dehazing system that combines multiple approaches
    for maximum accuracy and ground-truth-like results
    """
    
    def __init__(self, device='cpu'):
        self.device = torch.device(device) if isinstance(device, str) else device
        self.models = ['deep', 'enhanced', 'aod', 'natural', 'adaptive_natural', 'conservative', 'powerful']
        self.quality_weights = {}
        
    def calculate_image_quality_metrics(self, image: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive quality metrics for an image"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Normalize to 0-1 range
        gray_norm = gray.astype(np.float32) / 255.0
        
        # 1. Sharpness (Laplacian variance)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # 2. Contrast (standard deviation)
        contrast = np.std(gray_norm)
        
        # 3. Brightness appropriateness (distance from optimal 0.5)
        brightness = np.mean(gray_norm)
        brightness_score = 1.0 - abs(brightness - 0.5) * 2
        
        # 4. Color distribution (if color image)
        color_balance = 1.0
        if len(image.shape) == 3:
            b, g, r = cv2.split(image.astype(np.float32))
            mean_b, mean_g, mean_r = np.mean(b), np.mean(g), np.mean(r)
            # Penalize if one channel dominates too much
            channel_ratio = max(mean_r, mean_g, mean_b) / (min(mean_r, mean_g, mean_b) + 1e-6)
            color_balance = 1.0 / (1.0 + (channel_ratio - 1.0) * 0.1)
        
        # 5. Edge preservation
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # 6. Noise level (inverse of smoothness in non-edge areas)
        blurred = cv2.GaussianBlur(gray_norm, (5, 5), 1.0)
        noise_level = np.mean(np.abs(gray_norm - blurred))
        noise_score = max(0, 1.0 - noise_level * 10)  # Lower noise is better
        
        return {
            'sharpness': sharpness,
            'contrast': contrast,
            'brightness': brightness_score,
            'color_balance': color_balance,
            'edge_density': edge_density,
            'noise_score': noise_score
        }
    
    def calculate_overall_quality_score(self, metrics: Dict[str, float]) -> float:
        """Calculate weighted overall quality score"""
        weights = {
            'sharpness': 0.25,
            'contrast': 0.20,
            'brightness': 0.15,
            'color_balance': 0.20,
            'edge_density': 0.10,
            'noise_score': 0.10
        }
        
        # Normalize sharpness (typical range 0-1000)
        normalized_sharpness = min(1.0, metrics['sharpness'] / 500.0)
          # Normalize contrast (typical range 0-1)
        normalized_contrast = min(1.0, metrics['contrast'] * 2.0)
        
        # Normalize edge density (typical range 0-0.1)
        normalized_edge_density = min(1.0, metrics['edge_density'] * 10.0)
        
        score = (
            weights['sharpness'] * normalized_sharpness +
            weights['contrast'] * normalized_contrast +
            weights['brightness'] * metrics['brightness'] +
            weights['color_balance'] * metrics['color_balance'] +
            weights['edge_density'] * normalized_edge_density +
            weights['noise_score'] * metrics['noise_score']
        )
        
        return score
    
    def process_with_all_models(self, input_path: str, output_folder: str) -> Dict[str, Dict]:
        """Process image with all available models and return results with quality scores"""
        results = {}
        original_img = cv2.imread(input_path)
        
        if original_img is None:
            raise ValueError(f"Could not read image at {input_path}")
        
        logger.info("Processing with all models...")
        
        for model in self.models:
            try:
                start_time = time.time()
                
                # Process with current model
                if model in ['deep', 'enhanced', 'aod']:
                    result_path = process_image(input_path, output_folder, self.device, model)
                elif model == 'natural':
                    result_path = natural_dehaze(input_path, output_folder)
                elif model == 'adaptive_natural':
                    result_path = adaptive_natural_dehaze(input_path, output_folder)
                elif model == 'conservative':
                    result_path = conservative_color_dehaze(input_path, output_folder)
                elif model == 'powerful':
                    from powerful_dehazing import powerful_dehazing
                    result_path = powerful_dehazing(input_path, output_folder, strength='maximum')
                elif model == 'clahe':
                    result_path = dehaze_with_clahe(input_path, output_folder)
                elif model == 'crystal_clear':
                    from crystal_clear_model import process_image_crystal_clear
                    result_path = process_image_crystal_clear(input_path, output_folder)
                else:
                    continue
                
                processing_time = time.time() - start_time
                
                # Load result and calculate quality
                result_img = cv2.imread(result_path)
                if result_img is not None:
                    quality_metrics = self.calculate_image_quality_metrics(result_img)
                    overall_score = self.calculate_overall_quality_score(quality_metrics)
                    
                    results[model] = {
                        'path': result_path,
                        'image': result_img,
                        'processing_time': processing_time,
                        'quality_metrics': quality_metrics,
                        'quality_score': overall_score,
                        'success': True
                    }
                    
                    logger.info(f"{model}: Quality Score = {overall_score:.3f}, Time = {processing_time:.2f}s")
                
            except Exception as e:
                logger.error(f"Error processing with {model}: {str(e)}")
                results[model] = {
                    'success': False,
                    'error': str(e)
                }
        
        return results
    
    def smart_blend_results(self, results: Dict[str, Dict], blend_method: str = 'quality_weighted') -> np.ndarray:
        """Intelligently blend results from multiple models"""
        successful_results = {k: v for k, v in results.items() if v.get('success', False)}
        
        if not successful_results:
            raise ValueError("No successful processing results to blend")
        
        if len(successful_results) == 1:
            return list(successful_results.values())[0]['image']
        
        if blend_method == 'quality_weighted':
            return self._quality_weighted_blend(successful_results)
        elif blend_method == 'best_regions':
            return self._best_regions_blend(successful_results)
        else:
            return self._simple_average_blend(successful_results)
    
    def _quality_weighted_blend(self, results: Dict[str, Dict]) -> np.ndarray:
        """Blend images using quality scores as weights"""
        images = []
        weights = []
        
        for model, result in results.items():
            images.append(result['image'].astype(np.float32))
            weights.append(result['quality_score'])
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        
        # Weighted average
        blended = np.zeros_like(images[0])
        for img, weight in zip(images, weights):
            blended += img * weight
        
        return np.clip(blended, 0, 255).astype(np.uint8)
    
    def _best_regions_blend(self, results: Dict[str, Dict]) -> np.ndarray:
        """Blend by selecting best regions from each image"""
        images = list(results.values())
        if not images:
            raise ValueError("No images to blend")
        
        # For now, use quality weighted blend as placeholder
        # This could be enhanced with region-wise analysis
        return self._quality_weighted_blend(results)
    
    def _simple_average_blend(self, results: Dict[str, Dict]) -> np.ndarray:
        """Simple average blend of all results"""
        images = [result['image'].astype(np.float32) for result in results.values()]
        blended = np.mean(images, axis=0)
        return np.clip(blended, 0, 255).astype(np.uint8)
    
    def enhance_final_result(self, image: np.ndarray, enhancement_level: str = 'moderate') -> np.ndarray:
        """Apply final enhancement to the blended result"""
        if enhancement_level == 'none':
            return image

        enhanced = image.copy()

        if enhancement_level == 'ultra_strong':
            # BALANCED STRONG enhancement for clear visibility without over-processing

            # Step 1: STRONG but balanced CLAHE
            lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)

            # Strong but reasonable CLAHE parameters
            clahe = cv2.createCLAHE(clipLimit=3.2, tileGridSize=(7, 7))
            l_enhanced = clahe.apply(l)

            # Moderate color channel enhancement
            a = cv2.convertScaleAbs(a, alpha=1.15, beta=0)
            b = cv2.convertScaleAbs(b, alpha=1.15, beta=0)

            enhanced = cv2.merge([l_enhanced, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

            # Step 2: MODERATE contrast stretching
            for i in range(3):
                channel = enhanced[:,:,i].astype(np.float32)
                p_low, p_high = np.percentile(channel, [1.5, 98.5])  # Less aggressive
                if p_high > p_low:
                    channel = np.clip(255 * (channel - p_low) / (p_high - p_low), 0, 255)
                    enhanced[:,:,i] = channel.astype(np.uint8)

            # Step 3: MODERATE sharpening
            kernel_sharpen = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]) * 0.4
            sharpened = cv2.filter2D(enhanced, -1, kernel_sharpen)
            enhanced = cv2.addWeighted(enhanced, 0.8, sharpened, 0.2, 0)

            # Step 4: Gentle final enhancement
            enhanced = cv2.convertScaleAbs(enhanced, alpha=1.08, beta=3)

            return np.clip(enhanced, 0, 255).astype(np.uint8)

        # Original moderate/strong enhancement code
        if enhancement_level in ['moderate', 'strong']:
            # Apply adaptive histogram equalization
            lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)

            clip_limit = 2.0 if enhancement_level == 'moderate' else 3.0
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
            l_enhanced = clahe.apply(l)

            enhanced = cv2.merge([l_enhanced, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

        if enhancement_level == 'strong':
            # Additional sharpening for strong enhancement
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            enhanced = cv2.filter2D(enhanced, -1, kernel * 0.1)
            enhanced = cv2.addWeighted(image, 0.8, enhanced, 0.2, 0)

        return np.clip(enhanced, 0, 255).astype(np.uint8)
    
    def process_hybrid(self, input_path: str, output_folder: str, 
                      target_quality: float = 0.8, blend_method: str = 'quality_weighted',
                      enhancement_level: str = 'moderate') -> str:
        """
        Main hybrid processing function that combines all models intelligently
        
        Args:
            input_path: Path to input hazy image
            output_folder: Output directory
            target_quality: Target quality score (0.0 - 1.0)
            blend_method: 'quality_weighted', 'best_regions', or 'average'
            enhancement_level: 'none', 'moderate', or 'strong'
          Returns:
            Path to the final hybrid result        """
        logger.info(f"Starting hybrid dehazing for {input_path}")
        
        # Ensure output directory exists
        os.makedirs(output_folder, exist_ok=True)
        
        # Step 1: Process with all models
        all_results = self.process_with_all_models(input_path, output_folder)
        
        # Check if we have any successful results
        successful_models = [k for k, v in all_results.items() if v.get('success', False)]
        if not successful_models:
            raise ValueError("No models processed successfully. Please check your setup and input image.")
        
        # Step 2: Find best single result
        best_model = max(successful_models, key=lambda x: all_results[x]['quality_score'])
        best_score = all_results[best_model]['quality_score']
        
        logger.info(f"Best single model: {best_model} (score: {best_score:.3f})")
        
        # Step 3: Use intelligent blending based on quality
        if best_score >= target_quality:
            # Best single result is good enough
            final_result = all_results[best_model]['image']
            logger.info("Using best single result (quality target met)")
        else:
            # Blend multiple results for better quality
            final_result = self.smart_blend_results(all_results, blend_method)
            logger.info(f"Using blended result with method: {blend_method}")

        # Step 4: Apply BALANCED STRONG enhancement for clear visibility
        final_result = self.enhance_final_result(final_result, 'ultra_strong')
        
        # Step 5: Save final result
        base_filename = os.path.basename(input_path)
        filename, ext = os.path.splitext(base_filename)
        output_path = os.path.join(output_folder, f"{filename}_hybrid_dehazed{ext}")
        
        cv2.imwrite(output_path, final_result)
        
        # Calculate final quality score
        final_metrics = self.calculate_image_quality_metrics(final_result)
        final_score = self.calculate_overall_quality_score(final_metrics)
        
        logger.info(f"Hybrid dehazing completed. Final quality score: {final_score:.3f}")
        logger.info(f"Result saved to: {output_path}")
        
        return output_path

def process_hybrid_dehazing(input_path: str, output_folder: str, device='cpu', **kwargs) -> str:
    """Convenience function for hybrid dehazing"""
    ensemble = AdvancedDehazingEnsemble(device)
    return ensemble.process_hybrid(input_path, output_folder, **kwargs)
