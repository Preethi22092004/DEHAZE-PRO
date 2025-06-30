#!/usr/bin/env python3
"""
Test the web interface with perfect dehazing
"""

import requests
import os
import time

def test_perfect_dehazing():
    """Test the perfect dehazing through web interface"""
    
    # Test image path
    test_image = "test_images/playground_hazy.jpg"
    
    if not os.path.exists(test_image):
        print("❌ Test image not found")
        return False
    
    # Web interface URL
    url = "http://127.0.0.1:5000/upload-image"
    
    try:
        # Prepare the request
        with open(test_image, 'rb') as f:
            files = {'file': f}
            data = {'model': 'perfect'}  # Use perfect dehazing
            
            print("🚀 Testing perfect dehazing through web interface...")
            start_time = time.time()
            
            # Send request
            response = requests.post(url, files=files, data=data, timeout=30)
            
            processing_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                if result.get('success'):
                    print(f"✅ Perfect dehazing successful!")
                    print(f"⏱️  Processing time: {processing_time:.2f} seconds")
                    print(f"📁 Output: {result.get('output')}")
                    return True
                else:
                    print(f"❌ Processing failed: {result.get('error')}")
                    return False
            else:
                print(f"❌ HTTP Error: {response.status_code}")
                print(f"Response: {response.text}")
                return False
                
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to web interface. Make sure Flask server is running.")
        return False
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

if __name__ == '__main__':
    print("🧪 Testing Perfect Dehazing Web Interface")
    print("=" * 50)
    
    success = test_perfect_dehazing()
    
    if success:
        print("\n🎉 Perfect dehazing is working correctly!")
        print("✅ Web interface is ready for use")
        print("🌐 Visit http://127.0.0.1:5000 to try it")
    else:
        print("\n❌ Test failed. Please check the setup.")
