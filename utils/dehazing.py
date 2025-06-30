import os
import torch
import cv2
import numpy as np
from PIL import Image
import logging
from utils.model import load_model
from utils.direct_dehazing import natural_dehaze, adaptive_natural_dehaze, conservative_color_dehaze
from utils.minimal_dehazing import minimal_enhancement, no_processing
from powerful_dehazing import powerful_dehazing
from natural_balanced_dehazing import natural_balanced_dehazing
import time
import uuid
from torchvision import transforms

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variable to cache the models
_models = {}

def get_model(device, model_type='deep'):
    """Get or load the dehazing model."""
    global _models
    model_key = f"{model_type}_{device}"
    
    if model_key not in _models:
        logger.info(f"Loading {model_type} model on {device}")
        _models[model_key] = load_model(device, model_type)
    
    return _models[model_key]

def process_image(input_path, output_folder, device, model_type='deep'):
    """
    Process a single image using the dehazing model.

    Args:
        input_path (str): Path to the input hazy image
        output_folder (str): Directory to save the dehazed result
        device (torch.device or str): Device to run the model on
        model_type (str): Type of model to use (natural, adaptive_natural, conservative, deep, aod, light, enhanced, clahe)
        
    Returns:
        str: Path to the dehazed output image
    """
    # Check if this is a direct dehazing method (non-ML)
    if model_type in ['adaptive_natural', 'conservative', 'clahe', 'minimal', 'passthrough', 'natural_balanced', 'natural_gentle', 'natural_strong', 'powerful', 'powerful_high', 'powerful_moderate']:
        logger.info(f"Processing image with {model_type} direct dehazing method")
        
        # Apply the appropriate natural dehazing method
        if model_type == 'crystal_clear':
            from crystal_clear_model import process_image_crystal_clear
            return process_image_crystal_clear(input_path, output_folder)
        elif model_type == 'natural':
            # Use the best ML model for natural dehazing
            logger.info('Routing natural dehazing to DeepDehazeNet (ML model)')
            return process_image(input_path, output_folder, device, model_type='deep')
        elif model_type == 'adaptive_natural':
            return adaptive_natural_dehaze(input_path, output_folder)
        elif model_type == 'conservative':
            return conservative_color_dehaze(input_path, output_folder)
        elif model_type == 'powerful':
            # Use the powerful dehazing system with maximum strength
            return powerful_dehazing(input_path, output_folder, strength='maximum')
        elif model_type == 'powerful_high':
            return powerful_dehazing(input_path, output_folder, strength='high')
        elif model_type == 'powerful_moderate':
            return powerful_dehazing(input_path, output_folder, strength='moderate')
        elif model_type == 'natural_balanced':
            # Use the natural balanced dehazing system (recommended)
            return natural_balanced_dehazing(input_path, output_folder, strength='balanced')
        elif model_type == 'natural_gentle':
            return natural_balanced_dehazing(input_path, output_folder, strength='gentle')
        elif model_type == 'natural_strong':
            return natural_balanced_dehazing(input_path, output_folder, strength='strong')
        elif model_type == 'clahe':
            return dehaze_with_clahe(input_path, output_folder)
        elif model_type == 'minimal':
            return minimal_enhancement(input_path, output_folder)
        elif model_type == 'passthrough':
            return no_processing(input_path, output_folder)
    
    # For ML models, proceed with the original logic
    # Convert device to torch.device if it's a string
    if isinstance(device, str):
        device = torch.device(device)

    # Get the model
    model = get_model(device, model_type)
    
    # Load the image
    logger.info(f"Loading image from {input_path}")
    img = cv2.imread(input_path)
    if img is None:
        # Try different loading approach if standard method fails
        try:
            logger.warning(f"Standard image loading failed, trying with PIL for {input_path}")
            from PIL import Image
            pil_img = Image.open(input_path)
            img = np.array(pil_img.convert('RGB'))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # Convert from PIL's RGB to OpenCV's BGR
        except Exception as e:
            raise ValueError(f"Could not read image at {input_path}: {str(e)}")
    
    # Ensure the image has 3 channels (RGB)
    if len(img.shape) < 3:
        logger.warning(f"Converting grayscale image to RGB for {input_path}")
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:  # Has alpha channel
        logger.warning(f"Removing alpha channel from image {input_path}")
        img = img[:, :, :3]
    
    # Convert to RGB (from BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Normalize to [0, 1]
    img = img.astype(np.float32) / 255.0
    
    # Transform to tensor [C, H, W]
    img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(device)
    
    # Process with the model
    logger.info(f"Processing image with {model_type} dehazing model")
    with torch.no_grad():
        # Handle potential resize issues with DeepDehazeNet model
        if model_type == 'deep':
            # Ensure input dimensions are divisible by 8 for the deep model
            h, w = img_tensor.shape[2], img_tensor.shape[3]
            new_h = (h // 8) * 8
            new_w = (w // 8) * 8
            
            if h != new_h or w != new_w:
                logger.info(f"Resizing input from {h}x{w} to {new_h}x{new_w} for DeepDehazeNet")
                img_tensor = torch.nn.functional.interpolate(
                    img_tensor, size=(new_h, new_w), 
                    mode='bilinear', align_corners=False
                )
        
        # Run inference
        output = model(img_tensor)
        
        # Resize back to original dimensions if needed
        if model_type == 'deep' and (h != new_h or w != new_w):
            output = torch.nn.functional.interpolate(
                output, size=(h, w),
                mode='bilinear', align_corners=False
            )
    
    # Convert the output tensor back to numpy array
    output = output.squeeze().cpu().numpy().transpose(1, 2, 0)
    output = np.nan_to_num(output, nan=0.5, posinf=1.0, neginf=0.0)
    output = np.clip(output, 0, 1)
    output = (output * 255).astype(np.uint8)
    output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

    # === Post-processing for extra clarity ===
    # Gentle unsharp mask (sharpening)
    gaussian = cv2.GaussianBlur(output, (0, 0), 1.2)
    sharpened = cv2.addWeighted(output, 1.15, gaussian, -0.15, 0)
    # Gentle contrast enhancement (CLAHE on L channel)
    lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    final_output = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Generate output filename
    base_filename = os.path.basename(input_path)
    output_path = os.path.join(output_folder, base_filename)

    # Save the output image
    logger.info(f"Saving dehazed image to {output_path}")
    cv2.imwrite(output_path, final_output)

    return output_path

def process_video(input_path, output_folder, device, model_type='deep', frame_skip=1, max_resolution=720):
    """
    Process a video using the dehazing model with improved handling and optimization.
    
    Args:
        input_path (str): Path to the input hazy video
        output_folder (str): Directory to save the dehazed result
        device (torch.device): Device to run the model on
        model_type (str): Type of model to use
        frame_skip (int): Number of frames to skip between processing
        max_resolution (int): Maximum height resolution to process (wider dimension will be scaled proportionally)
        
    Returns:
        str: Path to the dehazed output video
    """
    # Get the model
    model = get_model(device, model_type)
    
    # Open the video file
    logger.info(f"Opening video from {input_path}")
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video at {input_path}")
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Resize video if too large to improve performance
    original_height = height
    original_width = width
    if height > max_resolution:
        # Calculate new dimensions while maintaining aspect ratio
        scale_factor = max_resolution / height
        width = int(width * scale_factor)
        height = max_resolution
        logger.info(f"Resizing video from {original_width}x{original_height} to {width}x{height} for faster processing")
    
    # Ensure even dimensions for video encoding
    width = width if width % 2 == 0 else width - 1
    height = height if height % 2 == 0 else height - 1
    
    # Create output filename
    base_filename = os.path.basename(input_path)
    filename, ext = os.path.splitext(base_filename)
    output_path = os.path.join(output_folder, f"{filename}_dehazed{ext}")
    
    # Choose appropriate codec based on output format and platform
    try:
        # Try to identify the best codec for the current platform
        if os.name == 'nt':  # Windows
            if ext.lower() in ['.mp4']:
                fourcc = cv2.VideoWriter_fourcc(*'H264')  # H.264 codec
                temp_ext = '.mp4'
            elif ext.lower() in ['.avi']:
                fourcc = cv2.VideoWriter_fourcc(*'XVID')  # XVID codec
                temp_ext = '.avi'
            else:
                # Default to XVID for unknown formats
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                temp_ext = '.avi'
        else:  # Linux/Mac
            if ext.lower() in ['.mp4']:
                fourcc = cv2.VideoWriter_fourcc(*'avc1')  # avc1 codec for mp4
                temp_ext = '.mp4'
            else:
                # Default to XVID for compatibility
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                temp_ext = '.avi'
    except Exception as e:
        logger.warning(f"Error selecting codec: {str(e)}. Using default XVID.")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        temp_ext = '.avi'
    
    # Create a temporary output path
    output_temp = os.path.join(output_folder, f"{filename}_temp{temp_ext}")
    
    # Initialize video writer with error handling
    try:
        out = cv2.VideoWriter(output_temp, fourcc, fps, (width, height))
        if not out.isOpened():
            logger.error(f"Failed to open video writer for {output_temp}")
            raise ValueError(f"Failed to open video writer for {output_temp}")
    except Exception as e:
        logger.error(f"Error creating video writer: {str(e)}")
        # Try with a different codec as fallback
        try:
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            output_temp = os.path.join(output_folder, f"{filename}_temp.avi")
            out = cv2.VideoWriter(output_temp, fourcc, fps, (width, height))
            if not out.isOpened():
                raise ValueError(f"Could not create video writer with fallback codec")
        except Exception as e2:
            logger.error(f"Failed with fallback codec: {str(e2)}")
            # Last resort - try with uncompressed codec
            fourcc = 0
            output_temp = os.path.join(output_folder, f"{filename}_temp.avi")
            out = cv2.VideoWriter(output_temp, fourcc, fps, (width, height))
            if not out.isOpened():
                raise ValueError("Could not create video writer with any codec")
        
    frame_idx = 0
    processed_frames = 0
    error_frames = 0
    start_time = time.time()
    max_consecutive_errors = 10  # Maximum allowed consecutive errors
    consecutive_errors = 0
    last_log_time = start_time
    
    logger.info(f"Starting video processing with {model_type} model")
    
    # Set up a progress tracking mechanism
    progress_interval = max(1, min(100, frame_count // 20))  # Log progress at ~5% intervals
    
    # Prepare a batch processing approach for better GPU utilization
    batch_size = 4 if device.type == 'cuda' and frame_skip == 1 else 1
    batch_frames = []
    
    try:
        while True:
            try:
                ret, frame = cap.read()
                if not ret:
                    # Process any remaining frames in the batch
                    if batch_frames and device.type == 'cuda':
                        process_batch(batch_frames, model, out, width, height)
                        processed_frames += len(batch_frames)
                        batch_frames = []
                    break
                    
                # Resize frame if needed to match output dimensions
                if original_height != height or original_width != width:
                    frame = cv2.resize(frame, (width, height))
                
                # Process every nth frame based on frame_skip
                if frame_idx % frame_skip == 0:
                    # Reset consecutive errors counter on successful frame read
                    consecutive_errors = 0
                    
                    # Batch processing for GPU if enabled
                    if device.type == 'cuda' and batch_size > 1:
                        # Convert to RGB and prepare tensor
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame_rgb = frame_rgb.astype(np.float32) / 255.0
                        frame_tensor = torch.from_numpy(frame_rgb.transpose(2, 0, 1)).unsqueeze(0)
                        batch_frames.append((frame_idx, frame, frame_tensor))
                        
                        # Process when batch is full
                        if len(batch_frames) >= batch_size:
                            process_batch(batch_frames, model, out, width, height, device)
                            processed_frames += len(batch_frames)
                            batch_frames = []
                    else:
                        # Process single frame
                        try:
                            # Convert to RGB (from BGR)
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            
                            # Normalize to [0, 1]
                            frame_rgb = frame_rgb.astype(np.float32) / 255.0
                            
                            # Transform to tensor [C, H, W]
                            frame_tensor = torch.from_numpy(frame_rgb.transpose(2, 0, 1)).unsqueeze(0).to(device)
                            
                            # Process with the model
                            with torch.no_grad():
                                output = model(frame_tensor)
                            
                            # Handle potential NaN values from the model
                            if torch.isnan(output).any() or torch.isinf(output).any():
                                logger.warning(f"NaN or Inf detected in output at frame {frame_idx}. Using original frame.")
                                out.write(frame)  # Use original frame as fallback
                            else:
                                # Convert the output tensor back to numpy array
                                output = output.squeeze().cpu().numpy().transpose(1, 2, 0)
                                
                                # Clip values to [0, 1] range
                                output = np.clip(output, 0, 1)
                                
                                # Convert to uint8
                                output = (output * 255).astype(np.uint8)
                                
                                # Convert back to BGR for OpenCV
                                output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
                                
                                # Write the frame
                                out.write(output_bgr)
                            
                            processed_frames += 1
                        except Exception as frame_error:
                            error_frames += 1
                            logger.warning(f"Error processing frame {frame_idx}: {str(frame_error)}. Using original frame.")
                            out.write(frame)
                else:
                    # Write the original frame for skipped frames
                    out.write(frame)
                
                # Log progress at intervals
                current_time = time.time()
                if processed_frames % progress_interval == 0 and current_time - last_log_time > 5:
                    elapsed = current_time - start_time
                    fps_processing = processed_frames / elapsed if elapsed > 0 else 0
                    progress = processed_frames / (frame_count / frame_skip) * 100 if frame_count > 0 else 0
                    logger.info(f"Processed {processed_frames} frames ({progress:.1f}%) at {fps_processing:.1f} fps")
                    last_log_time = current_time
                
                frame_idx += 1
                
                # Safety checks
                if frame_count > 0 and frame_idx > frame_count + 100:
                    logger.warning("Frame index exceeding expected frame count by large margin. Breaking.")
                    break
                    
            except Exception as read_error:
                consecutive_errors += 1
                logger.warning(f"Error reading frame: {str(read_error)}. Trying to continue...")
                
                if consecutive_errors >= max_consecutive_errors:
                    logger.error(f"Too many consecutive frame reading errors ({consecutive_errors}). Stopping processing.")
                    break
                
                frame_idx += 1
        
        # Log completion
        elapsed = time.time() - start_time
        fps_total = processed_frames / elapsed if elapsed > 0 else 0
        logger.info(f"Video processing complete. Processed {processed_frames} frames in {elapsed:.1f}s ({fps_total:.1f} fps)")
        logger.info(f"Error frames: {error_frames}")
    except Exception as e:
        logger.error(f"Error in video processing: {str(e)}")
    finally:
        # Clean up
        cap.release()
        out.release()
    
    # Return the temporary output path
    return output_temp

def process_batch(batch_frames, model, writer, width, height, device):
    """
    Process a batch of frames for more efficient GPU utilization.
    
    Args:
        batch_frames: List of tuples (frame_idx, original_frame, tensor_frame)
        model: The dehazing model
        writer: OpenCV VideoWriter object
        width, height: Output dimensions
        device: The processing device
    """
    if not batch_frames:
        return
    
    try:
        # Stack all tensors into a batch
        tensor_batch = torch.cat([f[2] for f in batch_frames], dim=0).to(device)
        
        # Process the entire batch at once
        with torch.no_grad():
            output_batch = model(tensor_batch)
        
        # Process each output separately
        for i, (frame_idx, original_frame, _) in enumerate(batch_frames):
            output = output_batch[i]
            
            # Check for NaN/Inf values
            if torch.isnan(output).any() or torch.isinf(output).any():
                writer.write(original_frame)
                continue
            
            # Convert to numpy and properly format
            output = output.cpu().numpy().transpose(1, 2, 0)
            output = np.clip(output, 0, 1)
            output = (output * 255).astype(np.uint8)
            output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
            
            # Write to video
            writer.write(output_bgr)
            
    except Exception as e:
        logger.error(f"Error in batch processing: {str(e)}")
        # Fallback to writing original frames
        for _, original_frame, _ in batch_frames:
            writer.write(original_frame)

def dehaze_with_clahe(image_path, output_folder):
    """
    Simple and effective CLAHE-based dehazing without color artifacts.

    Args:
        image_path (str): Path to the input hazy image
        output_folder (str): Directory to save the dehazed result

    Returns:
        str: Path to the dehazed output image
    """
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        from PIL import Image
        pil_img = Image.open(image_path)
        img = np.array(pil_img.convert('RGB'))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if img is None:
        raise ValueError(f"Could not read image at {image_path}")

    # Keep original for blending
    original = img.copy()
    
    # BALANCED STRONG CLAHE for clear visibility without over-processing
    # Convert to LAB color space for better color preservation
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    # Apply STRONG but balanced CLAHE to the L (lightness) channel
    clahe = cv2.createCLAHE(clipLimit=2.8, tileGridSize=(7, 7))  # Strong but not extreme
    l_enhanced = clahe.apply(l_channel)

    # Moderate color channel enhancement
    a_channel = cv2.convertScaleAbs(a_channel, alpha=1.1, beta=0)
    b_channel = cv2.convertScaleAbs(b_channel, alpha=1.1, beta=0)

    # Merge back the channels
    enhanced_lab = cv2.merge([l_enhanced, a_channel, b_channel])
    enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    # Apply MODERATE contrast stretching for clarity without artifacts
    for i in range(3):
        channel = enhanced_bgr[:,:,i].astype(np.float32)
        p_low, p_high = np.percentile(channel, [2, 98])  # Balanced percentiles
        if p_high > p_low:
            channel = np.clip(255 * (channel - p_low) / (p_high - p_low), 0, 255)
            enhanced_bgr[:,:,i] = channel.astype(np.uint8)

    # Apply gentle sharpening for clarity
    kernel_sharpen = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]) * 0.3  # Gentler
    sharpened = cv2.filter2D(enhanced_bgr, -1, kernel_sharpen)
    enhanced_bgr = cv2.addWeighted(enhanced_bgr, 0.8, sharpened, 0.2, 0)

    # BALANCED blending - strong enhancement but preserve natural look
    result = cv2.addWeighted(original, 0.25, enhanced_bgr, 0.75, 0)  # 75% enhanced, 25% original

    # Gentle final enhancement
    result = cv2.convertScaleAbs(result, alpha=1.05, beta=2)

    # Generate output filename
    base_filename = os.path.basename(image_path)
    filename, ext = os.path.splitext(base_filename)
    output_path = os.path.join(output_folder, f"{filename}_enhanced{ext}")

    # Save the result
    cv2.imwrite(output_path, result)

    return output_path

def dehaze_with_multiple_methods(input_path, output_folder, device):
    """
    Apply multiple dehazing methods to a single image and return paths to all results.
    
    Args:
        input_path (str): Path to the input hazy image
        output_folder (str): Directory to save the dehazed results
        device (torch.device): Device to run the models on
        
    Returns:
        dict: Paths to the dehazed output images from different methods
    """
    results = {}
    
    try:
        # Process with natural dehazing
        results['natural'] = process_image(input_path, output_folder, device, 'natural')
    except Exception as e:
        logger.error(f"Natural dehazing processing failed: {str(e)}")
    
    try:
        # Process with adaptive natural dehazing
        results['adaptive_natural'] = process_image(input_path, output_folder, device, 'adaptive_natural')
    except Exception as e:
        logger.error(f"Adaptive natural dehazing processing failed: {str(e)}")
    
    try:
        # Process with conservative dehazing
        results['conservative'] = process_image(input_path, output_folder, device, 'conservative')
    except Exception as e:
        logger.error(f"Conservative dehazing processing failed: {str(e)}")
    
    try:
        # Process with enhanced model
        results['enhanced'] = process_image(input_path, output_folder, device, 'enhanced')
    except Exception as e:
        logger.error(f"Enhanced model processing failed: {str(e)}")
    
    try:
        # Process with AOD-Net model
        results['aod'] = process_image(input_path, output_folder, device, 'aod')
    except Exception as e:
        logger.error(f"AOD-Net model processing failed: {str(e)}")
    
    try:
        # Process with CLAHE
        results['clahe'] = dehaze_with_clahe(input_path, output_folder)
    except Exception as e:
        logger.error(f"CLAHE processing failed: {str(e)}")
    
    if not results:
        raise ValueError("All dehazing methods failed")
    
    return results
