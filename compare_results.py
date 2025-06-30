#!/usr/bin/env python3
"""
Compare dehazing results
"""

import cv2
import numpy as np
import os
from pathlib import Path

def create_comparison_grid(images, titles, output_path):
    """Create a comparison grid of images"""
    if not images or len(images) != len(titles):
        raise ValueError("Images and titles must have the same length")
    
    # Resize all images to the same size
    target_height = 300
    resized_images = []
    
    for img in images:
        if img is not None:
            height, width = img.shape[:2]
            target_width = int(width * target_height / height)
            resized = cv2.resize(img, (target_width, target_height))
            resized_images.append(resized)
        else:
            # Create placeholder for missing image
            placeholder = np.ones((target_height, 400, 3), dtype=np.uint8) * 128
            cv2.putText(placeholder, "Image not found", (50, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            resized_images.append(placeholder)
    
    # Calculate grid dimensions
    n_images = len(resized_images)
    cols = min(3, n_images)  # Max 3 columns
    rows = (n_images + cols - 1) // cols
    
    # Get max dimensions
    max_width = max(img.shape[1] for img in resized_images)
    max_height = max(img.shape[0] for img in resized_images)
    
    # Create grid
    grid_width = cols * max_width
    grid_height = rows * (max_height + 50)  # Extra space for titles
    grid = np.ones((grid_height, grid_width, 3), dtype=np.uint8) * 255
    
    # Place images in grid
    for i, (img, title) in enumerate(zip(resized_images, titles)):
        row = i // cols
        col = i % cols
        
        y_start = row * (max_height + 50) + 30
        x_start = col * max_width + (max_width - img.shape[1]) // 2
        
        # Place image
        grid[y_start:y_start + img.shape[0], 
             x_start:x_start + img.shape[1]] = img
        
        # Add title
        title_y = row * (max_height + 50) + 20
        title_x = col * max_width + 10
        cv2.putText(grid, title, (title_x, title_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # Save comparison
    cv2.imwrite(output_path, grid)
    return output_path

def main():
    """Create comparison of dehazing results"""
    test_dir = Path("test_images")
    
    # Load images
    images = []
    titles = []
    
    # Original hazy image
    hazy_path = test_dir / "playground_hazy.jpg"
    if hazy_path.exists():
        images.append(cv2.imread(str(hazy_path)))
        titles.append("Original (Hazy)")
    
    # Dehazing results
    results = [
        ("playground_dehazed_hybrid.jpg", "Hybrid (Best)"),
        ("playground_dehazed_adaptive.jpg", "Adaptive Natural"),
        ("playground_dehazed_clahe.jpg", "CLAHE (Fast)"),
        ("playground_clear.jpg", "Reference Clear")
    ]
    
    for filename, title in results:
        img_path = test_dir / filename
        if img_path.exists():
            images.append(cv2.imread(str(img_path)))
            titles.append(title)
        else:
            images.append(None)
            titles.append(f"{title} (Missing)")
    
    # Create comparison
    if images:
        output_path = "dehazing_comparison.jpg"
        create_comparison_grid(images, titles, output_path)
        print(f"✅ Comparison saved to: {output_path}")
        
        # Print summary
        print("\n📊 Dehazing Results Summary:")
        print("=" * 50)
        for i, title in enumerate(titles):
            if images[i] is not None:
                print(f"✅ {title}")
            else:
                print(f"❌ {title}")
        
        print(f"\n🎯 Best method: Hybrid (combines multiple models)")
        print(f"⚡ Fastest method: CLAHE (traditional method)")
        print(f"🎨 Most natural: Adaptive Natural")
    else:
        print("❌ No images found to compare")

if __name__ == '__main__':
    main()
