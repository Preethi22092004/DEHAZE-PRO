#!/usr/bin/env python3
"""
Test script to verify web interface hybrid functionality
"""

import requests
import os
import time

def test_web_upload():
    """Test the web interface with hybrid processing"""
    
    url = "http://localhost:5000/upload-image"
    test_image = "test_hazy_image.jpg"
    
    if not os.path.exists(test_image):
        print("❌ Test image not found!")
        return False
    
    print("🌐 Testing web interface hybrid processing...")
    
    try:
        # Prepare the file upload
        with open(test_image, 'rb') as f:
            files = {'file': (test_image, f, 'image/jpeg')}
            data = {'model': 'hybrid'}
            
            print(f"📤 Uploading {test_image} with hybrid processing...")
            response = requests.post(url, files=files, data=data, timeout=120)
            
            if response.status_code == 200:
                result = response.json()
                if result.get('status') == 'success':
                    print("✅ Web interface hybrid processing successful!")
                    print(f"📁 Output file: {result.get('output_file')}")
                    print(f"🔧 Processing method: {result.get('processing_method')}")
                    print(f"⏱️ Processing time: {result.get('processing_time', 'N/A')}")
                    if 'quality_score' in result:
                        print(f"🏆 Quality score: {result.get('quality_score')}")
                    return True
                else:
                    print(f"❌ Processing failed: {result.get('message')}")
                    return False
            else:
                print(f"❌ HTTP Error: {response.status_code}")
                print(f"Response: {response.text}")
                return False
                
    except requests.exceptions.Timeout:
        print("⏰ Request timed out - processing may still be ongoing")
        return False
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to web server. Make sure it's running at http://localhost:5000")
        return False
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Hybrid Dehazing Web Interface")
    print("=" * 50)
    
    success = test_web_upload()
    
    if success:
        print("\n🎉 All tests passed! Web interface is working correctly.")
    else:
        print("\n⚠️ Test failed. Check the web server and try again.")
