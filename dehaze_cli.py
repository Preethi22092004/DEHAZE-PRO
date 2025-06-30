import argparse
import torch
import os
import sys
import time
from utils.dehazing import process_image, process_video, dehaze_with_clahe
from utils.hybrid_dehazing import process_hybrid_dehazing

def main():
    parser = argparse.ArgumentParser(description='Dehazing and Deobstruction System')
    parser.add_argument('--input', '-i', type=str, required=True, help='Path to input image or video')
    parser.add_argument('--output', '-o', type=str, help='Path to output directory (default: ./output)')
    parser.add_argument('--model', '-m', type=str, default='natural_balanced', 
                        choices=['natural_balanced', 'hybrid', 'natural', 'adaptive_natural', 'conservative', 'minimal', 'passthrough', 'enhanced', 'aod', 'deep', 'clahe', 'powerful'], 
                        help='Dehazing model to use (default: natural_balanced). Options: natural_balanced (recommended), hybrid, powerful, natural, adaptive_natural, conservative, minimal, passthrough, enhanced, aod, deep, clahe')
    parser.add_argument('--device', '-d', type=str, default='cpu', choices=['cuda', 'cpu'], 
                        help='Device to run the model on (default: cpu)')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = args.output if args.output else './output'
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' does not exist")
        sys.exit(1)
    
    # Check if CUDA is available when requested
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available. Using CPU instead.")
        device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    
    # Process input file
    start_time = time.time()
      # Check if input is image or video
    ext = os.path.splitext(args.input)[1].lower()
    if ext in ['.jpg', '.jpeg', '.png', '.webp']:
        # Process image
        print(f"Processing image with {args.model} model...")
        if args.model == 'hybrid':
            output_path = process_hybrid_dehazing(args.input, output_dir)
        elif args.model == 'clahe':
            output_path = dehaze_with_clahe(args.input, output_dir)
        else:
            output_path = process_image(args.input, output_dir, device, args.model)
    elif ext in ['.mp4', '.mov', '.avi', '.mkv']:
        # Process video
        print(f"Processing video with {args.model} model...")
        if args.model == 'hybrid':
            print("Warning: Hybrid processing for video not yet implemented. Using natural model instead.")
            output_path = process_video(args.input, output_dir, device, 'natural')
        else:
            output_path = process_video(args.input, output_dir, device, args.model)
    else:
        print(f"Error: Unsupported file format '{ext}'")
        sys.exit(1)
    
    # Print processing time
    processing_time = time.time() - start_time
    print(f"Processing completed in {processing_time:.2f} seconds")
    print(f"Output saved to {output_path}")

if __name__ == '__main__':
    main()
