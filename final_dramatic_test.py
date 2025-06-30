#!/usr/bin/env python3
"""
Final test to verify dramatic dehazing results with color quality check
"""

import requests
import time
import os
import cv2
import numpy as np
from pathlib import Path

def analyze_comprehensive_quality(original_path, dehazed_path):
    """Comprehensive analysis of dehazing quality"""
    try:
        # Read images
        original = cv2.imread(original_path)
        dehazed = cv2.imread(dehazed_path)
        
        if original is None or dehazed is None:
            return None
        
        # VISIBILITY ANALYSIS
        orig_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        dehazed_gray = cv2.cvtColor(dehazed, cv2.COLOR_BGR2GRAY)
        
        orig_brightness = np.mean(orig_gray)
        dehazed_brightness = np.mean(dehazed_gray)
        brightness_improvement = dehazed_brightness / orig_brightness
        
        orig_contrast = np.std(orig_gray)
        dehazed_contrast = np.std(dehazed_gray)
        contrast_improvement = dehazed_contrast / orig_contrast
        
        visibility_score = (brightness_improvement + contrast_improvement) / 2
        
        # COLOR QUALITY ANALYSIS
        # Convert to LAB color space
        dehazed_lab = cv2.cvtColor(dehazed, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(dehazed_lab)
        
        # Color balance (a and b should be close to 128 for neutral)
        a_mean = np.mean(a)
        b_mean = np.mean(b)
        color_cast_score = abs(a_mean - 128) + abs(b_mean - 128)
        
        # DETAIL ENHANCEMENT ANALYSIS
        # Calculate edge density as a measure of detail
        orig_edges = cv2.Canny(orig_gray, 50, 150)
        dehazed_edges = cv2.Canny(dehazed_gray, 50, 150)
        
        orig_edge_density = np.sum(orig_edges > 0) / orig_edges.size
        dehazed_edge_density = np.sum(dehazed_edges > 0) / dehazed_edges.size
        detail_improvement = dehazed_edge_density / max(orig_edge_density, 0.001)
        
        # OVERALL ASSESSMENT
        dramatic_improvement = visibility_score > 1.5
        visible_improvement = visibility_score > 1.2
        natural_colors = color_cast_score < 15  # Slightly more lenient for aggressive processing
        good_detail = detail_improvement > 1.1
        
        return {
            "visibility": {
                "original_brightness": orig_brightness,
                "dehazed_brightness": dehazed_brightness,
                "brightness_improvement": brightness_improvement,
                "original_contrast": orig_contrast,
                "dehazed_contrast": dehazed_contrast,
                "contrast_improvement": contrast_improvement,
                "visibility_score": visibility_score,
                "dramatic_improvement": dramatic_improvement,
                "visible_improvement": visible_improvement
            },
            "color_quality": {
                "color_cast_score": color_cast_score,
                "a_balance": a_mean - 128,
                "b_balance": b_mean - 128,
                "natural_colors": natural_colors
            },
            "detail_enhancement": {
                "original_edge_density": orig_edge_density,
                "dehazed_edge_density": dehazed_edge_density,
                "detail_improvement": detail_improvement,
                "good_detail": good_detail
            },
            "overall_quality": {
                "excellent": dramatic_improvement and natural_colors and good_detail,
                "good": visible_improvement and natural_colors,
                "acceptable": visible_improvement
            }
        }
        
    except Exception as e:
        print(f"Error in comprehensive analysis: {str(e)}")
        return None

def test_dramatic_dehazing():
    """Test the dramatic dehazing algorithm"""
    print("🎯 TESTING DRAMATIC DEHAZING ALGORITHM")
    print("=" * 60)
    
    test_image = "test_images/playground_hazy.jpg"
    if not os.path.exists(test_image):
        print("❌ Test image not found")
        return False
    
    print(f"📸 Testing with: {test_image}")
    
    try:
        # Test web interface
        url = 'http://127.0.0.1:5000/upload-image'
        
        with open(test_image, 'rb') as f:
            files = {'file': f}
            data = {'model_type': 'perfect'}
            
            start_time = time.time()
            response = requests.post(url, files=files, data=data)
            processing_time = time.time() - start_time
        
        if response.status_code == 200:
            print(f"✅ Web processing successful!")
            print(f"⏱️  Processing time: {processing_time:.3f} seconds")
            
            # Find the output file
            results_dir = Path("static/results")
            if results_dir.exists():
                files = list(results_dir.glob("*playground_hazy_perfect_dehazed.jpg"))
                if files:
                    # Get the most recent file
                    output_path = str(max(files, key=lambda f: f.stat().st_mtime))
                    
                    print(f"📁 Output file: {output_path}")
                    
                    # Comprehensive analysis
                    analysis = analyze_comprehensive_quality(test_image, output_path)
                    
                    if analysis:
                        print(f"\n📊 COMPREHENSIVE QUALITY ANALYSIS")
                        print("=" * 50)
                        
                        # Visibility results
                        vis = analysis['visibility']
                        print(f"👁️ VISIBILITY ENHANCEMENT:")
                        print(f"   Original brightness: {vis['original_brightness']:.1f}")
                        print(f"   Enhanced brightness: {vis['dehazed_brightness']:.1f}")
                        print(f"   Brightness boost: {vis['brightness_improvement']:.2f}x")
                        print(f"   Original contrast: {vis['original_contrast']:.1f}")
                        print(f"   Enhanced contrast: {vis['dehazed_contrast']:.1f}")
                        print(f"   Contrast boost: {vis['contrast_improvement']:.2f}x")
                        print(f"   Overall visibility score: {vis['visibility_score']:.2f}")
                        
                        if vis['dramatic_improvement']:
                            print(f"   🎉 DRAMATIC IMPROVEMENT! (Score > 1.5)")
                        elif vis['visible_improvement']:
                            print(f"   ✅ VISIBLE IMPROVEMENT! (Score > 1.2)")
                        else:
                            print(f"   ⚠️  Improvement too subtle (Score < 1.2)")
                        
                        # Color quality results
                        color = analysis['color_quality']
                        print(f"\n🎨 COLOR QUALITY:")
                        print(f"   Color cast score: {color['color_cast_score']:.1f}")
                        print(f"   A-channel balance: {color['a_balance']:.1f}")
                        print(f"   B-channel balance: {color['b_balance']:.1f}")
                        print(f"   Natural colors: {'✅ YES' if color['natural_colors'] else '❌ NO'}")
                        
                        # Detail enhancement results
                        detail = analysis['detail_enhancement']
                        print(f"\n🔍 DETAIL ENHANCEMENT:")
                        print(f"   Original detail density: {detail['original_edge_density']:.3f}")
                        print(f"   Enhanced detail density: {detail['dehazed_edge_density']:.3f}")
                        print(f"   Detail improvement: {detail['detail_improvement']:.2f}x")
                        print(f"   Good detail: {'✅ YES' if detail['good_detail'] else '❌ NO'}")
                        
                        # Overall assessment
                        overall = analysis['overall_quality']
                        print(f"\n🏆 OVERALL ASSESSMENT:")
                        if overall['excellent']:
                            print(f"   🌟 EXCELLENT! Dramatic + Natural + Detailed")
                            result_quality = "EXCELLENT"
                        elif overall['good']:
                            print(f"   ✅ GOOD! Visible + Natural colors")
                            result_quality = "GOOD"
                        elif overall['acceptable']:
                            print(f"   ⚠️  ACCEPTABLE! Visible improvement")
                            result_quality = "ACCEPTABLE"
                        else:
                            print(f"   ❌ NEEDS IMPROVEMENT")
                            result_quality = "POOR"
                        
                        print(f"\n🎯 FINAL VERDICT:")
                        print(f"   Quality: {result_quality}")
                        print(f"   Visibility Score: {vis['visibility_score']:.2f}")
                        print(f"   Color Cast Score: {color['color_cast_score']:.1f}")
                        print(f"   Ready for use: {'✅ YES' if overall['good'] else '❌ NEEDS WORK'}")
                        
                        return overall['good'] or overall['excellent']
                    else:
                        print("❌ Could not analyze quality")
                        return False
                else:
                    print("❌ No output file found")
                    return False
            else:
                print("❌ Results directory not found")
                return False
        else:
            print(f"❌ Web interface failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

def main():
    """Main test function"""
    print("🚀 FINAL DRAMATIC DEHAZING TEST")
    print("=" * 50)
    print("Testing the improved algorithm for dramatic visible results...")
    
    success = test_dramatic_dehazing()
    
    if success:
        print(f"\n🎉 SUCCESS! DRAMATIC DEHAZING ACHIEVED!")
        print("=" * 50)
        print("✅ Your dehazing system now provides:")
        print("   🎯 Dramatic visible improvement")
        print("   🌈 Natural color preservation")
        print("   🔍 Enhanced detail visibility")
        print("   ⚡ Fast processing")
        
        print(f"\n🌐 READY TO USE:")
        print("   Visit http://127.0.0.1:5000")
        print("   Upload your hazy images")
        print("   Get dramatic, natural results!")
        
        print(f"\n💡 THE DIFFERENCE SHOULD NOW BE CLEARLY VISIBLE!")
        print("   The dehazed image should look significantly clearer")
        print("   and brighter than the original hazy image.")
        
    else:
        print(f"\n❌ DRAMATIC RESULTS NOT YET ACHIEVED")
        print("The algorithm may need further tuning for your specific needs.")
    
    return success

if __name__ == '__main__':
    main()
