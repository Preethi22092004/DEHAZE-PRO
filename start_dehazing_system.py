#!/usr/bin/env python3
"""
Easy startup script for the dehazing system
"""

import os
import sys
import webbrowser
import time
import subprocess
from pathlib import Path

def check_requirements():
    """Check if all requirements are installed"""
    print("Checking system requirements...")

    try:
        import torch
        import cv2
        import flask
        from PIL import Image
        import numpy as np
        print("[OK] All required packages are installed")
        return True
    except ImportError as e:
        print(f"[FAIL] Missing required package: {e}")
        print("Please run: pip install -r requirements.txt")
        return False

def start_system():
    """Start the dehazing system"""
    print("STARTING DEHAZING SYSTEM")
    print("=" * 50)

    # Check requirements
    if not check_requirements():
        return False

    # Check if test passed
    print("\nRunning system test...")
    try:
        result = subprocess.run([sys.executable, "test_system_working.py"],
                              capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            print("[OK] System test passed!")
        else:
            print("[FAIL] System test failed!")
            print(result.stdout)
            print(result.stderr)
            return False
    except Exception as e:
        print(f"[FAIL] Error running system test: {e}")
        return False

    print("\nStarting web application...")
    print("The application will be available at: http://localhost:5000")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 50)
    
    # Wait a moment then open browser
    def open_browser():
        time.sleep(3)
        try:
            webbrowser.open('http://localhost:5000')
        except:
            pass
    
    import threading
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    # Start the Flask app
    try:
        os.system("python app.py")
    except KeyboardInterrupt:
        print("\n\nDehazing system stopped. Thank you for using it!")

    return True

if __name__ == "__main__":
    success = start_system()
    if not success:
        print("\n[FAIL] Failed to start the dehazing system")
        print("Please check the error messages above and try again")
        sys.exit(1)
