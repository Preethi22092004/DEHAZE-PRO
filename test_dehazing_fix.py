#!/usr/bin/env python3
"""
Test script to verify the dehazing fix is working correctly
"""

import os
import sys
import cv2
import numpy as np
import time
import shutil
from pathlib import Path

# Add current directory to path
sys.path.append('.')

from utils.perfect_trained_dehazing import perfect_trained_dehaze

def test_dehazing_fix():
    """Test the dehazing system with color correction"""
    
    print("=== DEHAZING FIX VERIFICATION ===")
    
    # Find a test image
    test_images = [
        'static/uploads/33669754-942b-414a-931e-14cad5abadf3blurr1.jpg',
        'test_hazy_image.jpg',
        'static/uploads/*.jpg'
    ]
    
    input_path = None
    for path in test_images:
        if os.path.exists(path):
            input_path = path
            break
    
    if not input_path:
        print("❌ No test image found. Please upload an image first.")
        return False
    
    print(f"✅ Using test image: {input_path}")
    
    # Test the dehazing
    try:
        print("🔄 Processing image with fixed dehazing...")
        result_path = perfect_trained_dehaze(input_path, 'static/results')
        
        if not os.path.exists(result_path):
            print("❌ Result image was not created")
            return False
        
        print(f"✅ Result saved to: {result_path}")
        
        # Analyze the result
        result_img = cv2.imread(result_path)
        if result_img is None:
            print("❌ Could not load result image")
            return False
        
        # Check color balance
        b, g, r = cv2.split(result_img)
        b_mean, g_mean, r_mean = np.mean(b), np.mean(g), np.mean(r)
        
        print(f"📊 Color Analysis:")
        print(f"   Blue:  {b_mean:.1f}")
        print(f"   Green: {g_mean:.1f}")
        print(f"   Red:   {r_mean:.1f}")
        
        # Check for purple tint
        blue_red_avg = (b_mean + r_mean) / 2
        has_purple_tint = (
            (blue_red_avg > g_mean * 1.4) or
            (b_mean > 150 and r_mean > 150 and g_mean < 100)
        )
        
        if has_purple_tint:
            print("❌ PURPLE TINT DETECTED - Fix not working properly")
            return False
        else:
            print("✅ Colors are properly balanced - NO PURPLE TINT!")
        
        # Create a direct access URL
        filename = os.path.basename(result_path)
        url = f"http://127.0.0.1:5000/results/{filename}"
        print(f"🌐 Direct URL: {url}")
        
        # Create a comparison HTML file
        create_comparison_html(input_path, result_path)
        
        return True
        
    except Exception as e:
        print(f"❌ Error during processing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def create_comparison_html(input_path, result_path):
    """Create an HTML file to display before/after comparison"""
    
    timestamp = int(time.time())
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Dehazing Fix Verification</title>
    <meta http-equiv='Cache-Control' content='no-cache, no-store, must-revalidate'>
    <meta http-equiv='Pragma' content='no-cache'>
    <meta http-equiv='Expires' content='0'>
    <style>
        body {{
            background: #1a1a1a;
            color: white;
            font-family: Arial, sans-serif;
            text-align: center;
            padding: 20px;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        .comparison {{
            display: flex;
            justify-content: space-around;
            align-items: flex-start;
            margin: 20px 0;
            flex-wrap: wrap;
        }}
        .image-box {{
            margin: 10px;
            padding: 15px;
            border: 2px solid #333;
            border-radius: 10px;
            background: #2a2a2a;
        }}
        .image-box img {{
            max-width: 500px;
            max-height: 400px;
            width: auto;
            height: auto;
            border: 1px solid #555;
        }}
        .image-title {{
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 10px;
            color: #4CAF50;
        }}
        .success {{
            background: #4CAF50;
            color: white;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
            font-size: 18px;
            font-weight: bold;
        }}
        .refresh-btn {{
            background: #2196F3;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
            margin: 10px;
        }}
        .refresh-btn:hover {{
            background: #1976D2;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🎉 DEHAZING FIX VERIFICATION</h1>
        
        <div class="success">
            ✅ Purple tint issue has been FIXED!<br>
            The dehazing system now produces natural, color-balanced results.
        </div>
        
        <div class="comparison">
            <div class="image-box">
                <div class="image-title">Original (Hazy)</div>
                <img src="/{input_path}?v={timestamp}" alt="Original Hazy Image">
            </div>
            
            <div class="image-box">
                <div class="image-title">Dehazed (Fixed)</div>
                <img src="/{result_path}?v={timestamp}" alt="Dehazed Image">
            </div>
        </div>
        
        <p>
            <button class="refresh-btn" onclick="location.reload()">🔄 Refresh</button>
            <button class="refresh-btn" onclick="window.open('/', '_blank')">🏠 Back to App</button>
        </p>
        
        <p style="color: #888; font-size: 14px;">
            Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}<br>
            Result file: {os.path.basename(result_path)}
        </p>
    </div>
</body>
</html>
    """
    
    html_path = f'templates/dehazing_fix_verification_{timestamp}.html'
    with open(html_path, 'w') as f:
        f.write(html_content)
    
    print(f"📄 Comparison page created: {html_path}")
    print(f"🌐 View at: http://127.0.0.1:5000/view_result/dehazing_fix_verification_{timestamp}")

if __name__ == "__main__":
    success = test_dehazing_fix()
    if success:
        print("\n🎉 DEHAZING FIX VERIFICATION SUCCESSFUL!")
        print("The purple tint issue has been resolved.")
    else:
        print("\n❌ DEHAZING FIX VERIFICATION FAILED!")
        print("The issue may still exist.")
