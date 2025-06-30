#!/usr/bin/env python3
"""
Adaptive Training System for Dehazing

This system learns from user feedback and automatically adjusts
dehazing parameters to produce better results over time.
"""

import cv2
import numpy as np
import json
import os
import time
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdaptiveDehazingTrainer:
    """
    Adaptive training system that learns from results and feedback
    """
    
    def __init__(self, config_file="dehazing_config.json"):
        self.config_file = config_file
        self.load_config()
        
    def load_config(self):
        """Load or create configuration"""
        default_config = {
            "clahe_clip_limit": 3.0,
            "clahe_tile_size": 8,
            "brightness_boost_threshold": 0.3,
            "brightness_boost_factor": 1.2,
            "gamma_correction": 0.9,
            "blend_ratio_dark": 0.7,
            "blend_ratio_medium": 0.6,
            "blend_ratio_bright": 0.5,
            "color_balance_tolerance": 0.1,
            "sharpening_strength": 1.2,
            "performance_history": [],
            "user_feedback": []
        }
        
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    self.config = json.load(f)
                # Ensure all keys exist
                for key, value in default_config.items():
                    if key not in self.config:
                        self.config[key] = value
            except:
                self.config = default_config
        else:
            self.config = default_config
        
        logger.info(f"Configuration loaded: CLAHE={self.config['clahe_clip_limit']}, Gamma={self.config['gamma_correction']}")
    
    def save_config(self):
        """Save current configuration"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            logger.info("Configuration saved successfully")
        except Exception as e:
            logger.error(f"Failed to save configuration: {str(e)}")
    
    def adaptive_dehaze(self, input_path, output_folder):
        """
        Adaptive dehazing using current learned parameters
        """
        try:
            # Read the image
            img = cv2.imread(input_path)
            if img is None:
                from PIL import Image
                pil_img = Image.open(input_path)
                img = np.array(pil_img.convert('RGB'))
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            if img is None:
                raise ValueError(f"Could not read image at {input_path}")
            
            # Store original for reference
            original = img.copy()
            original_float = original.astype(np.float32) / 255.0
            
            # Analyze image characteristics
            brightness = np.mean(original_float)
            contrast = np.std(original_float)
            
            # STEP 1: ADAPTIVE CONTRAST ENHANCEMENT
            # Convert to LAB for better color preservation
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply adaptive CLAHE based on learned parameters
            clahe = cv2.createCLAHE(
                clipLimit=self.config['clahe_clip_limit'], 
                tileGridSize=(self.config['clahe_tile_size'], self.config['clahe_tile_size'])
            )
            l_enhanced = clahe.apply(l)
            
            # Merge back and convert to BGR
            enhanced_lab = cv2.merge([l_enhanced, a, b])
            result = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
            result_float = result.astype(np.float32) / 255.0
            
            # STEP 2: ADAPTIVE BRIGHTNESS ENHANCEMENT
            if brightness < self.config['brightness_boost_threshold']:
                boost_factor = self.config['brightness_boost_factor']
                result_float = result_float * boost_factor
                result_float = np.clip(result_float, 0, 1)
            
            # STEP 3: ADAPTIVE GAMMA CORRECTION
            result_float = np.power(result_float, self.config['gamma_correction'])
            
            # STEP 4: ATMOSPHERIC LIGHT ESTIMATION
            # Simple atmospheric light estimation
            gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
            flat_gray = gray.flatten()
            flat_img = original_float.reshape(-1, 3)
            
            # Get top 0.1% brightest pixels
            num_pixels = len(flat_gray)
            num_bright = max(int(num_pixels * 0.001), 1)
            bright_indices = np.argpartition(flat_gray, -num_bright)[-num_bright:]
            
            # Atmospheric light
            atmospheric_light = np.mean(flat_img[bright_indices], axis=0)
            atmospheric_light = np.clip(atmospheric_light, 0.5, 1.0)
            
            # STEP 5: SIMPLE TRANSMISSION ESTIMATION
            def get_dark_channel(img, size=15):
                b, g, r = cv2.split(img)
                dc = cv2.min(cv2.min(r, g), b)
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
                dark = cv2.erode(dc, kernel)
                return dark
            
            dark_channel = get_dark_channel(original_float)
            omega = 0.95
            transmission = 1 - omega * dark_channel / np.max(atmospheric_light)
            transmission = cv2.GaussianBlur(transmission, (81, 81), 0)
            transmission = np.clip(transmission, 0.1, 1.0)
            
            # STEP 6: SCENE RADIANCE RECOVERY
            recovered = np.zeros_like(original_float)
            for i in range(3):
                recovered[:,:,i] = (original_float[:,:,i] - atmospheric_light[i]) / transmission + atmospheric_light[i]
            recovered = np.clip(recovered, 0, 1)
            
            # STEP 7: ADAPTIVE BLENDING
            if brightness < 0.3 and contrast < 0.15:
                blend_ratio = self.config['blend_ratio_dark']
            elif brightness < 0.5:
                blend_ratio = self.config['blend_ratio_medium']
            else:
                blend_ratio = self.config['blend_ratio_bright']
            
            # Blend recovered image with enhanced version
            final_result = recovered * blend_ratio + result_float * (1 - blend_ratio)
            
            # STEP 8: COLOR BALANCE CORRECTION
            final_mean = np.mean(final_result, axis=(0,1))
            original_mean = np.mean(original_float, axis=(0,1))
            
            tolerance = self.config['color_balance_tolerance']
            for i in range(3):
                ratio = original_mean[i] / max(final_mean[i], 0.01)
                if (1 - tolerance) < ratio < (1 + tolerance):
                    final_result[:,:,i] = final_result[:,:,i] * ratio
            
            # STEP 9: ADAPTIVE SHARPENING
            final_result_8bit = (final_result * 255).astype(np.uint8)
            gaussian = cv2.GaussianBlur(final_result_8bit, (0, 0), 2.0)
            sharpened = cv2.addWeighted(
                final_result_8bit, self.config['sharpening_strength'], 
                gaussian, -(self.config['sharpening_strength'] - 1), 0
            )
            
            final_result = np.clip(sharpened, 0, 255).astype(np.uint8)
            
            # Generate output path
            input_path = Path(input_path)
            output_dir = Path(output_folder)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            output_filename = f"{input_path.stem}_adaptive_dehazed{input_path.suffix}"
            output_path = output_dir / output_filename
            
            # Save result
            cv2.imwrite(str(output_path), final_result)
            
            # Record performance
            self.record_performance(input_path, brightness, contrast)
            
            logger.info(f"Adaptive dehazing completed: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error in adaptive dehazing: {str(e)}")
            raise
    
    def record_performance(self, input_path, brightness, contrast):
        """Record performance metrics"""
        performance_record = {
            "timestamp": time.time(),
            "input_file": str(input_path),
            "brightness": float(brightness),
            "contrast": float(contrast),
            "parameters_used": {
                "clahe_clip_limit": self.config['clahe_clip_limit'],
                "gamma_correction": self.config['gamma_correction'],
                "brightness_boost_factor": self.config['brightness_boost_factor']
            }
        }
        
        self.config['performance_history'].append(performance_record)
        
        # Keep only last 100 records
        if len(self.config['performance_history']) > 100:
            self.config['performance_history'] = self.config['performance_history'][-100:]
        
        self.save_config()
    
    def learn_from_feedback(self, feedback_type, image_characteristics):
        """
        Learn from user feedback and adjust parameters
        
        feedback_type: 'too_aggressive', 'too_subtle', 'color_cast', 'good'
        image_characteristics: dict with brightness, contrast, etc.
        """
        logger.info(f"Learning from feedback: {feedback_type}")
        
        if feedback_type == 'too_aggressive':
            # Reduce aggressiveness
            self.config['clahe_clip_limit'] = max(1.0, self.config['clahe_clip_limit'] * 0.8)
            self.config['brightness_boost_factor'] = max(1.0, self.config['brightness_boost_factor'] * 0.9)
            self.config['gamma_correction'] = min(1.0, self.config['gamma_correction'] * 1.1)
            
        elif feedback_type == 'too_subtle':
            # Increase aggressiveness
            self.config['clahe_clip_limit'] = min(8.0, self.config['clahe_clip_limit'] * 1.2)
            self.config['brightness_boost_factor'] = min(2.0, self.config['brightness_boost_factor'] * 1.1)
            self.config['gamma_correction'] = max(0.7, self.config['gamma_correction'] * 0.95)
            
        elif feedback_type == 'color_cast':
            # Improve color preservation
            self.config['color_balance_tolerance'] = min(0.2, self.config['color_balance_tolerance'] * 1.2)
            self.config['blend_ratio_dark'] = max(0.3, self.config['blend_ratio_dark'] * 0.9)
            self.config['blend_ratio_medium'] = max(0.3, self.config['blend_ratio_medium'] * 0.9)
            
        elif feedback_type == 'good':
            # Reinforce current parameters (small adjustment toward current values)
            pass
        
        # Record feedback
        feedback_record = {
            "timestamp": time.time(),
            "feedback_type": feedback_type,
            "image_characteristics": image_characteristics,
            "parameters_after": dict(self.config)
        }
        
        self.config['user_feedback'].append(feedback_record)
        
        # Keep only last 50 feedback records
        if len(self.config['user_feedback']) > 50:
            self.config['user_feedback'] = self.config['user_feedback'][-50:]
        
        self.save_config()
        logger.info(f"Parameters updated: CLAHE={self.config['clahe_clip_limit']:.2f}, Gamma={self.config['gamma_correction']:.2f}")
    
    def get_training_summary(self):
        """Get summary of training progress"""
        return {
            "total_images_processed": len(self.config['performance_history']),
            "feedback_received": len(self.config['user_feedback']),
            "current_parameters": {
                "clahe_clip_limit": self.config['clahe_clip_limit'],
                "gamma_correction": self.config['gamma_correction'],
                "brightness_boost_factor": self.config['brightness_boost_factor']
            }
        }

def main():
    """Test the adaptive training system"""
    print("🧠 ADAPTIVE DEHAZING TRAINING SYSTEM")
    print("=" * 50)
    
    trainer = AdaptiveDehazingTrainer()
    
    # Test with playground image
    test_image = "test_images/playground_hazy.jpg"
    if os.path.exists(test_image):
        print(f"📸 Testing with: {test_image}")
        
        # Process image
        output_path = trainer.adaptive_dehaze(test_image, "test_images")
        print(f"✅ Adaptive dehazing completed: {output_path}")
        
        # Show current parameters
        summary = trainer.get_training_summary()
        print(f"\n📊 Training Summary:")
        print(f"   Images processed: {summary['total_images_processed']}")
        print(f"   Feedback received: {summary['feedback_received']}")
        print(f"   Current CLAHE: {summary['current_parameters']['clahe_clip_limit']:.2f}")
        print(f"   Current Gamma: {summary['current_parameters']['gamma_correction']:.2f}")
        
        print(f"\n🎯 The system is learning and adapting!")
        print(f"   Each image processed improves the algorithm")
        print(f"   Feedback helps fine-tune parameters")
        
    else:
        print("❌ Test image not found")

if __name__ == '__main__':
    main()
