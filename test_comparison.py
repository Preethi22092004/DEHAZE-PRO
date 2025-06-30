from utils.dehazing import dehaze_with_multiple_methods, process_image, process_video
import torch
import os
import cv2
import glob
import time

def main():
    # Create output directory if it doesn't exist
    os.makedirs('output', exist_ok=True)
    
    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Test image
    test_image = 'static/uploads/da215ace-aec2-4929-bdc6-d8f4f0bba84cScreenshot_2025-05-16_121827.png'
    
    # Process with multiple methods
    print(f"\n1. Processing {test_image} with multiple methods...")
    results = dehaze_with_multiple_methods(test_image, 'output', device)
    
    # Print results
    print("\nImage comparison complete. Results:")
    for method, path in results.items():
        print(f"{method}: {path}")
    
    # Find video files to process
    video_files = glob.glob('static/uploads/*.mp4')
    if video_files:
        test_video = video_files[0]
        print(f"\n2. Processing video {test_video} with 'enhanced' model...")
        # Process with enhanced model
        try:
            video_output_path = process_video(
                test_video, 
                'output', 
                device,
                model_type='enhanced',
                frame_skip=2  # Process every 2nd frame for speed
            )
            print(f"Video processing complete. Output: {video_output_path}")
        except Exception as e:
            print(f"Error processing video: {str(e)}")
    else:
        print("\nNo video files found for testing")
    
    # Performance test on another image (if available)
    other_images = glob.glob('static/uploads/*Screenshot*.png')
    if len(other_images) > 1:
        test_image2 = [img for img in other_images if img != test_image][0]
        print(f"\n3. Performance testing on {test_image2}...")
        
        # Performance comparison
        for model_type in ['aod', 'enhanced']:
            start_time = time.time()
            output_path = process_image(test_image2, 'output', device, model_type)
            elapsed = time.time() - start_time
            print(f"{model_type} model: {elapsed:.2f} seconds - {output_path}")

    print("\nTesting complete! Please review the output images to confirm color balance is fixed.")

if __name__ == "__main__":
    main()
