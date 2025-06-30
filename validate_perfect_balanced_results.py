"""
Perfect Balanced Results Validation
==================================

This script validates the results from the perfectly balanced dehazing model
to ensure it meets your quality requirements:
- Crystal clear like your reference image
- Natural colors (not too aggressive)
- Professional quality (not too simple)
"""

import os
import cv2
import numpy as np
import logging
from pathlib import Path
import json
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QualityValidator:
    """Validates dehazing quality against reference standards"""
    
    def __init__(self, reference_image_path=None):
        self.reference_image_path = reference_image_path
        self.reference_features = None
        
        if reference_image_path and os.path.exists(reference_image_path):
            self.reference_features = self.extract_reference_features(reference_image_path)
    
    def extract_reference_features(self, image_path):
        """Extract quality features from reference image"""
        
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
        
        features = {
            'brightness': np.mean(image),
            'contrast': np.std(image),
            'color_balance': np.mean(image, axis=(0, 1)),
            'edge_strength': self.calculate_edge_strength(image),
            'clarity_level': self.calculate_clarity_level(image),
            'naturalness_score': self.calculate_naturalness_score(image)
        }
        
        return features
    
    def calculate_edge_strength(self, image):
        """Calculate edge strength (clarity indicator)"""
        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        return np.mean(edges) / 255.0
    
    def calculate_clarity_level(self, image):
        """Calculate clarity level using Laplacian variance"""
        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        clarity = np.var(laplacian) / 10000.0
        return min(clarity, 1.0)
    
    def calculate_naturalness_score(self, image):
        """Calculate naturalness score (lower = more natural)"""
        
        # Color distribution naturalness
        color_std = np.std(image, axis=(0, 1))
        color_naturalness = 1.0 - np.mean(np.abs(color_std - 0.15))  # Natural std around 0.15
        
        # Brightness naturalness
        brightness = np.mean(image)
        brightness_naturalness = 1.0 - abs(brightness - 0.5)  # Natural brightness around 0.5
        
        # Overall naturalness
        naturalness = (color_naturalness + brightness_naturalness) / 2
        return max(0, min(1, naturalness))
    
    def calculate_aggressiveness_score(self, original, processed):
        """Calculate aggressiveness score (higher = more aggressive)"""
        
        # Brightness change
        orig_brightness = np.mean(original)
        proc_brightness = np.mean(processed)
        brightness_change = abs(proc_brightness - orig_brightness)
        
        # Contrast change
        orig_contrast = np.std(original)
        proc_contrast = np.std(processed)
        contrast_change = abs(proc_contrast - orig_contrast)
        
        # Color saturation change
        orig_saturation = np.mean(np.std(original, axis=(0, 1)))
        proc_saturation = np.mean(np.std(processed, axis=(0, 1)))
        saturation_change = abs(proc_saturation - orig_saturation)
        
        # Combined aggressiveness
        aggressiveness = (brightness_change + contrast_change + saturation_change) / 3
        return min(1.0, aggressiveness * 2)  # Scale to 0-1
    
    def validate_image_quality(self, original_path, processed_path):
        """Validate the quality of a processed image"""
        
        # Load images
        original = cv2.imread(original_path)
        processed = cv2.imread(processed_path)
        
        if original is None or processed is None:
            logger.error(f"Could not load images: {original_path}, {processed_path}")
            return None
        
        # Convert to RGB and normalize
        original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        
        # Resize to same size if needed
        if original.shape != processed.shape:
            processed = cv2.resize(processed, (original.shape[1], original.shape[0]))
        
        # Extract features
        processed_features = {
            'brightness': np.mean(processed),
            'contrast': np.std(processed),
            'color_balance': np.mean(processed, axis=(0, 1)),
            'edge_strength': self.calculate_edge_strength(processed),
            'clarity_level': self.calculate_clarity_level(processed),
            'naturalness_score': self.calculate_naturalness_score(processed)
        }
        
        # Calculate quality metrics
        metrics = {
            'clarity_improvement': processed_features['clarity_level'] - self.calculate_clarity_level(original),
            'edge_enhancement': processed_features['edge_strength'] - self.calculate_edge_strength(original),
            'brightness_change': abs(processed_features['brightness'] - np.mean(original)),
            'contrast_change': abs(processed_features['contrast'] - np.std(original)),
            'naturalness_score': processed_features['naturalness_score'],
            'aggressiveness_score': self.calculate_aggressiveness_score(original, processed)
        }
        
        # Reference comparison if available
        if self.reference_features:
            metrics['reference_similarity'] = self.calculate_reference_similarity(processed_features)
        
        # Overall quality assessment
        metrics['overall_quality'] = self.calculate_overall_quality(metrics)
        
        return metrics, processed_features
    
    def calculate_reference_similarity(self, processed_features):
        """Calculate similarity to reference image"""
        
        if not self.reference_features:
            return 0.5
        
        # Compare key features
        brightness_sim = 1 - abs(processed_features['brightness'] - self.reference_features['brightness'])
        contrast_sim = 1 - abs(processed_features['contrast'] - self.reference_features['contrast'])
        clarity_sim = 1 - abs(processed_features['clarity_level'] - self.reference_features['clarity_level'])
        edge_sim = 1 - abs(processed_features['edge_strength'] - self.reference_features['edge_strength'])
        
        # Weighted similarity
        similarity = (
            brightness_sim * 0.25 +
            contrast_sim * 0.25 +
            clarity_sim * 0.25 +
            edge_sim * 0.25
        )
        
        return max(0, min(1, similarity))
    
    def calculate_overall_quality(self, metrics):
        """Calculate overall quality score"""
        
        # Positive factors
        clarity_score = min(1.0, metrics['clarity_improvement'] * 2)  # Clarity improvement
        naturalness_score = metrics['naturalness_score']  # Natural appearance
        
        # Negative factors (penalties)
        aggressiveness_penalty = metrics['aggressiveness_score']  # Too aggressive
        
        # Reference similarity bonus
        reference_bonus = metrics.get('reference_similarity', 0.5)
        
        # Combined score
        quality_score = (
            clarity_score * 0.3 +
            naturalness_score * 0.3 +
            reference_bonus * 0.2 +
            (1 - aggressiveness_penalty) * 0.2
        )
        
        return max(0, min(1, quality_score))
    
    def assess_quality_level(self, quality_score):
        """Assess quality level based on score"""
        
        if quality_score >= 0.85:
            return "EXCELLENT", "✅ Perfect balanced quality achieved!"
        elif quality_score >= 0.75:
            return "VERY_GOOD", "✅ High quality with good balance"
        elif quality_score >= 0.65:
            return "GOOD", "✅ Good quality, acceptable balance"
        elif quality_score >= 0.55:
            return "ACCEPTABLE", "⚠️  Acceptable quality, could be improved"
        elif quality_score >= 0.45:
            return "NEEDS_IMPROVEMENT", "⚠️  Quality needs improvement"
        else:
            return "POOR", "❌ Poor quality, significant issues"

def validate_results():
    """Validate the perfect balanced dehazing results"""
    
    logger.info("Perfect Balanced Dehazing Results Validation")
    logger.info("="*60)
    
    # Initialize validator with reference image
    reference_images = [
        'test_images/playground_hazy.jpg',
        'test_hazy_image.jpg'
    ]
    
    reference_image = None
    for img_path in reference_images:
        if os.path.exists(img_path):
            reference_image = img_path
            break
    
    validator = QualityValidator(reference_image)
    logger.info(f"Using reference image: {reference_image}")
    
    # Test cases - Updated to include final results
    test_cases = [
        {
            'name': 'Playground Image Test - Final Model',
            'original': 'test_images/playground_hazy.jpg',
            'processed': 'test_results/playground_hazy_final_perfect_balanced.jpg'
        },
        {
            'name': 'Test Hazy Image - Final Model',
            'original': 'test_hazy_image.jpg',
            'processed': 'test_results/test_hazy_image_final_perfect_balanced.jpg'
        },
        {
            'name': 'Playground Image Test - Quick Model',
            'original': 'test_images/playground_hazy.jpg',
            'processed': 'test_results/playground_hazy_quick_perfect_balanced.jpg'
        },
        {
            'name': 'Test Hazy Image - Quick Model',
            'original': 'test_hazy_image.jpg',
            'processed': 'test_results/test_hazy_image_quick_perfect_balanced.jpg'
        }
    ]
    
    results = []
    
    for test_case in test_cases:
        if not os.path.exists(test_case['original']) or not os.path.exists(test_case['processed']):
            logger.warning(f"Skipping {test_case['name']}: files not found")
            continue
        
        logger.info(f"\nValidating: {test_case['name']}")
        logger.info("-" * 40)
        
        # Validate quality
        validation_result = validator.validate_image_quality(
            test_case['original'], 
            test_case['processed']
        )
        
        if validation_result is None:
            continue
        
        metrics, features = validation_result
        
        # Assess quality
        quality_level, quality_message = validator.assess_quality_level(metrics['overall_quality'])
        
        # Log results
        logger.info(f"Overall Quality Score: {metrics['overall_quality']:.3f}")
        logger.info(f"Quality Level: {quality_level}")
        logger.info(f"Assessment: {quality_message}")
        logger.info(f"Clarity Improvement: {metrics['clarity_improvement']:.3f}")
        logger.info(f"Naturalness Score: {metrics['naturalness_score']:.3f}")
        logger.info(f"Aggressiveness Score: {metrics['aggressiveness_score']:.3f}")
        
        if 'reference_similarity' in metrics:
            logger.info(f"Reference Similarity: {metrics['reference_similarity']:.3f}")
        
        # Store results
        results.append({
            'test_case': test_case['name'],
            'quality_score': metrics['overall_quality'],
            'quality_level': quality_level,
            'metrics': metrics,
            'features': features
        })
    
    # Overall assessment
    if results:
        avg_quality = np.mean([r['quality_score'] for r in results])
        logger.info(f"\n{'='*60}")
        logger.info(f"OVERALL ASSESSMENT")
        logger.info(f"{'='*60}")
        logger.info(f"Average Quality Score: {avg_quality:.3f}")
        
        overall_level, overall_message = validator.assess_quality_level(avg_quality)
        logger.info(f"Overall Quality Level: {overall_level}")
        logger.info(f"Final Assessment: {overall_message}")
        
        # Recommendations
        logger.info(f"\nRECOMMENDations:")
        if avg_quality >= 0.8:
            logger.info("✅ Model is performing excellently!")
            logger.info("✅ Perfect balance achieved between clarity and naturalness")
            logger.info("✅ Ready for production use")
        elif avg_quality >= 0.7:
            logger.info("✅ Model is performing well")
            logger.info("⚠️  Minor improvements possible")
        else:
            logger.info("⚠️  Model needs further training")
            logger.info("⚠️  Consider adjusting training parameters")
        
        # Save validation report
        report = {
            'validation_date': datetime.now().isoformat(),
            'reference_image': reference_image,
            'average_quality_score': avg_quality,
            'overall_quality_level': overall_level,
            'test_results': results
        }
        
        os.makedirs('validation_reports', exist_ok=True)
        report_path = 'validation_reports/perfect_balanced_validation_report.json'
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"\nValidation report saved: {report_path}")
    
    return results

if __name__ == "__main__":
    validate_results()
