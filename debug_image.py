import requests
import os
import cv2
import numpy as np

# Check if the Flask server is running and serving the image correctly
try:
    response = requests.get('http://127.0.0.1:5000/static/results/33669754-942b-414a-931e-14cad5abadf3blurr1_perfect_trained_dehazed.jpg')
    print(f'HTTP Status: {response.status_code}')
    print(f'Content-Type: {response.headers.get("Content-Type", "Not set")}')
    print(f'Content-Length: {len(response.content)} bytes')
    
    # Check if the content matches the file
    file_path = 'static/results/33669754-942b-414a-931e-14cad5abadf3blurr1_perfect_trained_dehazed.jpg'
    with open(file_path, 'rb') as f:
        file_content = f.read()
    
    print(f'File size: {len(file_content)} bytes')
    print(f'Content matches file: {response.content == file_content}')
    
    # Also check the image visually
    img = cv2.imread(file_path)
    print(f'Image shape: {img.shape}')
    print(f'Image data type: {img.dtype}')
    print(f'Image min/max: {np.min(img)}/{np.max(img)}')
    
    # Save a copy with a different name to test browser caching
    cv2.imwrite('static/results/test_copy.jpg', img)
    print('Saved test copy as test_copy.jpg')
    
except Exception as e:
    print(f'Error: {e}')
