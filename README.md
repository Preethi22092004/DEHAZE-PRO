# Dehazing and Deobstruction System

This project is a complete Dehazing and Deobstruction System that can take images or videos affected by fog, smoke, water, haze, blur, or other obstructions and output a clear, original-looking version without generating artificial content.

## Features

- Process both static images and videos
- Multiple dehazing algorithms:
  - AOD-Net (All-in-One Dehazing Network)
  - Enhanced Dehazing (LightDehazeNet)
  - CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Web interface for easy use
- Mobile-friendly responsive design
- PWA support for installation on mobile devices
- API endpoints for integration with other applications
- Real-time processing capabilities
- Database storage for tracking processed media

## System Requirements

- Python 3.8 or higher
- PyTorch (CPU or GPU version)
- OpenCV
- Flask
- Other dependencies listed in requirements.txt

## Installation

1. Clone the repository or download the source code.

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Generate model weights (if not already present):

```bash
python generate_weights.py
```

## Running the Application

The simplest way to run the application is:

```bash
python app.py
```

The application will be available at http://localhost:5000

## Accessing from Mobile Devices

Once the application is running, you can access it from any mobile device on the same network:

1. Find your computer's IP address:
   - On Windows: Run `ipconfig` in the command prompt
   - On macOS/Linux: Run `ifconfig` or `ip addr show`

2. Access the application on your mobile device by entering the IP address and port in the browser:
   - Example: `http://192.168.0.215:5000` (replace with your actual IP address)

3. For a better mobile experience, you can install the app as a PWA:
   - On Android: Open the site in Chrome, tap the menu button, and select "Add to Home Screen"
   - On iOS: Open the site in Safari, tap the Share button, and select "Add to Home Screen"

## Usage

1. **Image Dehazing**:
   - Navigate to the home page (http://localhost:5000)
   - Upload an image using the "Upload Image" button
   - Select a dehazing method (Enhanced, AOD-Net, or CLAHE)
   - View and download the result

2. **Video Dehazing**:
   - Navigate to the video page (http://localhost:5000/video)
   - Upload a video file
   - Wait for processing to complete (this may take some time depending on the video length)
   - View and download the processed video

3. **Compare Methods**:
   - Navigate to the compare page (http://localhost:5000/compare)
   - Upload an image
   - View results from all available dehazing methods side by side

## Command Line Interface

For processing images and videos via command line:

```bash
python dehaze_cli.py --input "path\to\image.jpg" --model enhanced
```

## API Endpoints

The system includes the following API endpoints:

- `/upload-image` - For processing images
- `/upload-video` - For processing videos
- `/video-status/<task_id>` - For checking video processing status
- `/api/models` - For listing available dehazing models

## File Structure

- `app.py` - Main Flask application
- `main.py` - Entry point for the application
- `models.py` - Database models
- `utils/` - Utility functions
  - `dehazing.py` - Image and video processing functions
  - `model.py` - Neural network model definitions
- `static/` - Static files (CSS, JS, etc.)
- `templates/` - HTML templates
- `generate_weights.py` - Script for generating placeholder model weights

## Future Improvements

- Training on additional datasets for better performance
- Support for more dehazing models
- Mobile app development (React Native or Flutter)
- Performance optimizations for real-time processing
- Support for ONNX/TensorFlow Lite for on-device processing

## License

[MIT License](LICENSE)
