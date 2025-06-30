#!/usr/bin/env python3
"""
Simple Dehazing CLI Tool
========================

A straightforward command-line tool for removing fog, haze, smoke, and blur from images
while preserving original image details without AI-generated content replacement.

Usage:
    python simple_dehaze.py input_image.jpg [output_image.jpg] [--method METHOD]

Methods available:
    - perfect (default): Optimized single-step dehazing with perfect color balance
    - hybrid: Best quality using ensemble of multiple models
    - natural: Fast natural dehazing preserving colors
    - deep: AI-based deep learning model
    - clahe: Fast traditional method
    - enhanced: Enhanced AI model
    - aod: AOD-Net model
"""

import argparse
import os
import sys
import time
import torch
import logging
from pathlib import Path

# Add the current directory to Python path to import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.dehazing import process_image, dehaze_with_clahe
from utils.hybrid_dehazing import process_hybrid_dehazing
from utils.perfect_dehazing import perfect_dehaze, simple_perfect_dehaze, ultra_safe_dehaze
from utils.maximum_dehazing import maximum_strength_dehaze

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def setup_device():
    """Setup the best available device (GPU if available, otherwise CPU)"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU (GPU not available)")
    return device

def validate_input_file(input_path):
    """Validate that the input file exists and is a supported image format"""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    file_ext = Path(input_path).suffix.lower()
    
    if file_ext not in supported_formats:
        raise ValueError(f"Unsupported file format: {file_ext}. Supported formats: {', '.join(supported_formats)}")
    
    return True

def generate_output_path(input_path, output_path=None, method='perfect'):
    """Generate output path if not provided"""
    if output_path:
        return output_path
    
    input_path = Path(input_path)
    output_dir = input_path.parent
    stem = input_path.stem
    suffix = input_path.suffix
    
    return str(output_dir / f"{stem}_dehazed_{method}{suffix}")

def dehaze_image(input_path, output_path, method='perfect', device=None):
    """
    Main dehazing function that processes the image with the specified method
    
    Args:
        input_path (str): Path to input hazy image
        output_path (str): Path for output dehazed image
        method (str): Dehazing method to use
        device: PyTorch device to use for processing
    
    Returns:
        str: Path to the processed image
    """
    if device is None:
        device = setup_device()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    logger.info(f"Processing: {input_path}")
    logger.info(f"Method: {method}")
    logger.info(f"Output: {output_path}")
    
    start_time = time.time()
    
    try:
        if method == 'perfect':
            # Use the maximum strength dehazing for crystal clear results
            result_path = maximum_strength_dehaze(input_path, os.path.dirname(output_path))
            # Move to desired output path
            if result_path != output_path:
                import shutil
                shutil.move(result_path, output_path)
                result_path = output_path
        elif method == 'hybrid':
            # Use the advanced hybrid ensemble system for best results
            result_path = process_hybrid_dehazing(
                input_path,
                os.path.dirname(output_path),
                device=device,
                target_quality=0.8,
                blend_method='quality_weighted',
                enhancement_level='moderate'
            )
            # Move to desired output path
            if result_path != output_path:
                import shutil
                shutil.move(result_path, output_path)
                result_path = output_path
                
        elif method == 'clahe':
            # Use fast CLAHE method
            result_path = dehaze_with_clahe(input_path, os.path.dirname(output_path))
            # Move to desired output path
            if result_path != output_path:
                import shutil
                shutil.move(result_path, output_path)
                result_path = output_path
                
        else:
            # Use one of the ML models
            result_path = process_image(
                input_path, 
                os.path.dirname(output_path), 
                device, 
                model_type=method
            )
            # Move to desired output path
            if result_path != output_path:
                import shutil
                shutil.move(result_path, output_path)
                result_path = output_path
        
        processing_time = time.time() - start_time
        logger.info(f"✅ Processing completed in {processing_time:.2f} seconds")
        logger.info(f"✅ Dehazed image saved to: {result_path}")
        
        return result_path
        
    except Exception as e:
        logger.error(f"❌ Error processing image: {str(e)}")
        raise

def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(
        description="Remove fog, haze, smoke, and blur from images while preserving original details",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python simple_dehaze.py hazy_image.jpg
  python simple_dehaze.py hazy_image.jpg clear_image.jpg
  python simple_dehaze.py hazy_image.jpg --method natural
  python simple_dehaze.py hazy_image.jpg clear_image.jpg --method deep
        """
    )
    
    parser.add_argument('input', help='Path to the hazy input image')
    parser.add_argument('output', nargs='?', help='Path for the dehazed output image (optional)')
    parser.add_argument('--method', '-m',
                       choices=['perfect', 'hybrid', 'natural', 'deep', 'clahe', 'enhanced', 'aod', 'adaptive_natural', 'conservative'],
                       default='perfect',
                       help='Dehazing method to use (default: perfect)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Validate input
        validate_input_file(args.input)
        
        # Generate output path if not provided
        output_path = generate_output_path(args.input, args.output, args.method)
        
        # Setup device
        device = setup_device()
        
        # Process the image
        result_path = dehaze_image(args.input, output_path, args.method, device)
        
        print(f"\n🎉 Success! Dehazed image saved to: {result_path}")
        
    except KeyboardInterrupt:
        logger.info("\n⏹️  Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ Error: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()
