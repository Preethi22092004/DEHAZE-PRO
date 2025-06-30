# Simple Dehazing Tool - Usage Guide

## Overview

This tool removes fog, haze, smoke, and blur from images while preserving original image details **without AI-generated content replacement**. It processes your images to reveal the original content that was obscured by atmospheric conditions.

## Quick Start

### Basic Usage
```bash
# Process with best quality (hybrid method)
python simple_dehaze.py your_hazy_image.jpg

# Specify output filename
python simple_dehaze.py input.jpg output_clear.jpg

# Use specific method
python simple_dehaze.py input.jpg --method clahe
```

## Available Methods

### 🏆 **hybrid** (Recommended - Best Quality)
- **Description**: Combines multiple AI models intelligently
- **Quality**: Highest (automatically selects best approach)
- **Speed**: Medium (1-2 seconds)
- **Best for**: When you want the absolute best results

### ⚡ **clahe** (Fastest)
- **Description**: Traditional contrast enhancement method
- **Quality**: Good for most images
- **Speed**: Very fast (<1 second)
- **Best for**: Quick processing, batch operations

### 🎨 **adaptive_natural** (Most Natural)
- **Description**: Preserves natural colors and appearance
- **Quality**: High with natural look
- **Speed**: Very fast (<1 second)
- **Best for**: Photos where natural appearance is critical

### 🧠 **deep** (AI-Powered)
- **Description**: Deep learning dehazing network
- **Quality**: Very high
- **Speed**: Medium (1 second)
- **Best for**: Complex haze patterns

### 🔧 **enhanced** (AI Enhanced)
- **Description**: ResNet-based enhancement model
- **Quality**: High
- **Speed**: Fast (<1 second)
- **Best for**: General purpose dehazing

### 🌐 **aod** (AOD-Net)
- **Description**: All-in-One Dehazing Network
- **Quality**: High
- **Speed**: Fast (<1 second)
- **Best for**: Fog and atmospheric haze

## Examples

### Process Your Playground Images
```bash
# Best quality result
python simple_dehaze.py playground_foggy.jpg playground_clear.jpg --method hybrid

# Fast processing
python simple_dehaze.py playground_foggy.jpg playground_clear.jpg --method clahe

# Natural appearance
python simple_dehaze.py playground_foggy.jpg playground_clear.jpg --method adaptive_natural
```

### Batch Processing
```bash
# Process multiple images
for img in *.jpg; do
    python simple_dehaze.py "$img" "dehazed_$img" --method hybrid
done
```

## Performance Comparison

Based on our testing with playground images:

| Method | Quality Score | Speed | Best Use Case |
|--------|---------------|-------|---------------|
| **hybrid** | 0.816 | Medium | Best overall results |
| **deep** | 0.816 | Medium | AI-powered dehazing |
| **adaptive_natural** | 0.808 | Fast | Natural appearance |
| **aod** | 0.801 | Fast | Atmospheric conditions |
| **clahe** | Good | Very Fast | Quick enhancement |

## Tips for Best Results

### 1. **Choose the Right Method**
- **For best quality**: Use `hybrid` (default)
- **For speed**: Use `clahe` or `adaptive_natural`
- **For natural look**: Use `adaptive_natural`

### 2. **Input Image Quality**
- Higher resolution images generally produce better results
- Supported formats: JPG, PNG, BMP, TIFF, WebP

### 3. **Expected Results**
- ✅ Removes atmospheric haze, fog, smoke
- ✅ Enhances visibility and contrast
- ✅ Preserves original image details
- ✅ No AI-generated content replacement
- ✅ Maintains natural colors and textures

## Troubleshooting

### Common Issues

**"Input file not found"**
- Check the file path is correct
- Ensure the image file exists

**"Unsupported file format"**
- Convert to JPG, PNG, BMP, TIFF, or WebP

**Slow processing**
- Use `clahe` or `adaptive_natural` for faster results
- GPU acceleration available if CUDA is installed

### Getting Help
```bash
python simple_dehaze.py --help
```

## System Requirements

- Python 3.7+
- OpenCV
- PyTorch
- NumPy
- PIL/Pillow

## Installation

If you encounter missing dependencies:
```bash
pip install torch torchvision opencv-python pillow numpy scipy scikit-image
```

---

## Example Results

The tool successfully processes images like your playground example:
- **Input**: Foggy/hazy playground image
- **Output**: Clear, detailed playground image with fog removed
- **Preservation**: All original playground equipment, colors, and details maintained
- **No AI Generation**: Only removes obstructions, doesn't create new content

Perfect for removing atmospheric conditions while keeping your original image intact!
