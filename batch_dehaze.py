#!/usr/bin/env python3
"""
Batch Dehazing Tool
==================

Process multiple images at once with the dehazing system.

Usage:
    python batch_dehaze.py input_folder/ output_folder/ [--method METHOD]
    python batch_dehaze.py *.jpg --output-dir results/ [--method METHOD]
"""

import argparse
import os
import sys
import time
import glob
from pathlib import Path
import logging

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from simple_dehaze import dehaze_image, setup_device, validate_input_file

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def find_images(input_path):
    """Find all image files in the input path"""
    supported_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.webp']
    
    images = []
    
    if os.path.isdir(input_path):
        # Search directory for images
        for ext in supported_extensions:
            pattern = os.path.join(input_path, '**', ext)
            images.extend(glob.glob(pattern, recursive=True))
            # Also search for uppercase extensions
            pattern = os.path.join(input_path, '**', ext.upper())
            images.extend(glob.glob(pattern, recursive=True))
    else:
        # Treat as glob pattern
        images = glob.glob(input_path)
    
    # Remove duplicates and sort
    images = sorted(list(set(images)))
    
    # Filter valid images
    valid_images = []
    for img in images:
        try:
            validate_input_file(img)
            valid_images.append(img)
        except Exception as e:
            logger.warning(f"Skipping {img}: {str(e)}")
    
    return valid_images

def generate_output_path(input_path, output_dir, method):
    """Generate output path for processed image"""
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate output filename
    stem = input_path.stem
    suffix = input_path.suffix
    output_filename = f"{stem}_dehazed_{method}{suffix}"
    
    return str(output_dir / output_filename)

def process_batch(input_images, output_dir, method='hybrid', device=None):
    """Process a batch of images"""
    if not input_images:
        logger.error("No valid images found to process")
        return []
    
    if device is None:
        device = setup_device()
    
    logger.info(f"Processing {len(input_images)} images with {method} method")
    logger.info(f"Output directory: {output_dir}")
    
    results = []
    failed = []
    total_time = 0
    
    for i, input_path in enumerate(input_images, 1):
        try:
            logger.info(f"[{i}/{len(input_images)}] Processing: {os.path.basename(input_path)}")
            
            # Generate output path
            output_path = generate_output_path(input_path, output_dir, method)
            
            # Process image
            start_time = time.time()
            result_path = dehaze_image(input_path, output_path, method, device)
            processing_time = time.time() - start_time
            total_time += processing_time
            
            results.append({
                'input': input_path,
                'output': result_path,
                'time': processing_time,
                'success': True
            })
            
            logger.info(f"✅ Completed in {processing_time:.2f}s")
            
        except Exception as e:
            logger.error(f"❌ Failed to process {input_path}: {str(e)}")
            failed.append({
                'input': input_path,
                'error': str(e),
                'success': False
            })
    
    # Print summary
    print(f"\n📊 Batch Processing Summary")
    print("=" * 50)
    print(f"✅ Successfully processed: {len(results)} images")
    print(f"❌ Failed: {len(failed)} images")
    print(f"⏱️  Total time: {total_time:.2f} seconds")
    if results:
        print(f"⚡ Average time per image: {total_time/len(results):.2f} seconds")
    
    if failed:
        print(f"\n❌ Failed images:")
        for item in failed:
            print(f"   - {os.path.basename(item['input'])}: {item['error']}")
    
    return results + failed

def main():
    """Main batch processing function"""
    parser = argparse.ArgumentParser(
        description="Batch process multiple images to remove fog, haze, smoke, and blur",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python batch_dehaze.py photos/ results/
  python batch_dehaze.py photos/ results/ --method clahe
  python batch_dehaze.py "*.jpg" --output-dir dehazed/
  python batch_dehaze.py input_folder/ output_folder/ --method hybrid --verbose
        """
    )
    
    parser.add_argument('input', help='Input directory or glob pattern (e.g., "*.jpg")')
    parser.add_argument('output', nargs='?', help='Output directory')
    parser.add_argument('--output-dir', '-o', help='Output directory (alternative to positional argument)')
    parser.add_argument('--method', '-m',
                       choices=['perfect', 'hybrid', 'natural', 'deep', 'clahe', 'enhanced', 'aod', 'adaptive_natural', 'conservative'],
                       default='perfect',
                       help='Dehazing method to use (default: perfect)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Determine output directory
    output_dir = args.output or args.output_dir
    if not output_dir:
        output_dir = 'dehazed_results'
        logger.info(f"No output directory specified, using: {output_dir}")
    
    try:
        # Find input images
        logger.info(f"Searching for images in: {args.input}")
        input_images = find_images(args.input)
        
        if not input_images:
            logger.error(f"No valid images found in: {args.input}")
            sys.exit(1)
        
        logger.info(f"Found {len(input_images)} images to process")
        
        # Setup device
        device = setup_device()
        
        # Process batch
        results = process_batch(input_images, output_dir, args.method, device)
        
        # Final success message
        successful = [r for r in results if r.get('success', False)]
        if successful:
            print(f"\n🎉 Batch processing completed!")
            print(f"📁 Results saved in: {output_dir}")
        else:
            print(f"\n❌ No images were processed successfully")
            sys.exit(1)
        
    except KeyboardInterrupt:
        logger.info("\n⏹️  Batch processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ Error: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()
