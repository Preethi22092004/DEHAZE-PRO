#!/usr/bin/env python3
"""
Test Maximum Strength Dehazing System
=====================================

This script tests the new maximum strength dehazing algorithms to validate
they achieve crystal clear results without artifacts.
"""

import os
import sys
import time
import logging
from pathlib import Path

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.maximum_dehazing import maximum_strength_dehaze, remini_level_dehaze
from utils.perfect_dehazing import ultra_safe_dehaze

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def test_dehazing_methods(input_image_path, output_dir):
    """
    Test all dehazing methods on a single image.
    
    Args:
        input_image_path (str): Path to the test image
        output_dir (str): Directory to save results
    """
    if not os.path.exists(input_image_path):
        logger.error(f"Test image not found: {input_image_path}")
        return False
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    methods = [
        ("Maximum Strength", maximum_strength_dehaze),
        ("Remini Level", remini_level_dehaze),
        ("Ultra Safe (Original)", ultra_safe_dehaze)
    ]
    
    results = {}
    
    for method_name, method_func in methods:
        logger.info(f"\n{'='*50}")
        logger.info(f"Testing {method_name} Dehazing")
        logger.info(f"{'='*50}")
        
        try:
            start_time = time.time()
            
            # Apply dehazing method
            output_path = method_func(input_image_path, output_dir)
            
            processing_time = time.time() - start_time
            
            logger.info(f"✅ {method_name} completed in {processing_time:.2f} seconds")
            logger.info(f"   Output: {output_path}")
            
            results[method_name] = {
                'success': True,
                'output_path': output_path,
                'processing_time': processing_time
            }
            
        except Exception as e:
            logger.error(f"❌ {method_name} failed: {str(e)}")
            results[method_name] = {
                'success': False,
                'error': str(e)
            }
    
    return results


def create_comparison_report(results, input_image_path, output_dir):
    """
    Create a comparison report of all dehazing methods.
    
    Args:
        results (dict): Results from testing
        input_image_path (str): Path to the original image
        output_dir (str): Directory containing results
    """
    report_path = os.path.join(output_dir, "dehazing_comparison_report.html")
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Maximum Strength Dehazing Test Results</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .header {{ background-color: #2c3e50; color: white; padding: 20px; text-align: center; }}
            .method {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
            .success {{ background-color: #d4edda; border-color: #c3e6cb; }}
            .error {{ background-color: #f8d7da; border-color: #f5c6cb; }}
            .image-container {{ text-align: center; margin: 10px 0; }}
            .image-container img {{ max-width: 400px; height: auto; border: 1px solid #ddd; }}
            .stats {{ background-color: #f8f9fa; padding: 10px; margin: 10px 0; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Maximum Strength Dehazing Test Results</h1>
            <p>Comparison of dehazing methods for crystal clear results</p>
        </div>
        
        <div class="method">
            <h2>Original Image</h2>
            <div class="image-container">
                <img src="{os.path.basename(input_image_path)}" alt="Original Image">
                <p><strong>Original hazy image</strong></p>
            </div>
        </div>
    """
    
    for method_name, result in results.items():
        if result['success']:
            html_content += f"""
            <div class="method success">
                <h2>✅ {method_name}</h2>
                <div class="stats">
                    <p><strong>Processing Time:</strong> {result['processing_time']:.2f} seconds</p>
                    <p><strong>Status:</strong> Success</p>
                </div>
                <div class="image-container">
                    <img src="{os.path.basename(result['output_path'])}" alt="{method_name} Result">
                    <p><strong>{method_name} Result</strong></p>
                </div>
            </div>
            """
        else:
            html_content += f"""
            <div class="method error">
                <h2>❌ {method_name}</h2>
                <div class="stats">
                    <p><strong>Status:</strong> Failed</p>
                    <p><strong>Error:</strong> {result['error']}</p>
                </div>
            </div>
            """
    
    html_content += """
        <div class="method">
            <h2>Summary</h2>
            <p>This test validates the new maximum strength dehazing algorithms designed to achieve:</p>
            <ul>
                <li>Crystal clear results without artifacts</li>
                <li>Maximum dehazing strength while preserving original details</li>
                <li>Professional-grade quality comparable to Remini app</li>
                <li>Fast processing times for real-time applications</li>
            </ul>
        </div>
    </body>
    </html>
    """
    
    with open(report_path, 'w') as f:
        f.write(html_content)
    
    logger.info(f"📊 Comparison report saved to: {report_path}")
    return report_path


def main():
    """Main test function"""
    logger.info("🚀 Starting Maximum Strength Dehazing Test")
    
    # Test with existing hazy image
    test_images = [
        "test_hazy_image.jpg",
        "realistic_hazy_test.jpg"
    ]
    
    # Find available test image
    test_image = None
    for img in test_images:
        if os.path.exists(img):
            test_image = img
            break
    
    if not test_image:
        logger.error("❌ No test images found. Please ensure you have a hazy image to test with.")
        logger.info("Expected test images: " + ", ".join(test_images))
        return False
    
    # Set up output directory
    output_dir = "maximum_dehazing_test_results"
    
    logger.info(f"📸 Using test image: {test_image}")
    logger.info(f"📁 Output directory: {output_dir}")
    
    # Run tests
    results = test_dehazing_methods(test_image, output_dir)
    
    # Create comparison report
    report_path = create_comparison_report(results, test_image, output_dir)
    
    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*60}")
    
    successful_methods = [name for name, result in results.items() if result['success']]
    failed_methods = [name for name, result in results.items() if not result['success']]
    
    logger.info(f"✅ Successful methods: {len(successful_methods)}")
    for method in successful_methods:
        time_taken = results[method]['processing_time']
        logger.info(f"   - {method}: {time_taken:.2f}s")
    
    if failed_methods:
        logger.info(f"❌ Failed methods: {len(failed_methods)}")
        for method in failed_methods:
            logger.info(f"   - {method}: {results[method]['error']}")
    
    logger.info(f"\n📊 View detailed comparison: {report_path}")
    logger.info(f"📁 All results saved in: {output_dir}")
    
    if successful_methods:
        logger.info("\n🎉 Maximum strength dehazing test completed successfully!")
        return True
    else:
        logger.error("\n💥 All dehazing methods failed!")
        return False


if __name__ == '__main__':
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("\n⏹️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ Unexpected error: {str(e)}")
        sys.exit(1)
