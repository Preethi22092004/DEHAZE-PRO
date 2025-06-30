# DeepDehazeNet Model Guide

## Introduction

The DeepDehazeNet model is now the default advanced dehazing model in the system. This guide explains how to use it and its advantages over previous models.

## Features

The DeepDehazeNet model combines multiple state-of-the-art techniques:

1. **Dense Feature Extraction with Dilated Convolutions** - Captures wider context without losing resolution
2. **Multi-scale Feature Fusion** - Combines features at different scales for better dehazing
3. **Attention-guided Refinement** - Focuses on hazy regions while preserving clean regions
4. **Enhanced Transmission Map Estimation** - More accurate for various types of haze
5. **Adaptive Contrast Enhancement** - Improves visual quality in the final output

## When to Use DeepDehazeNet

DeepDehazeNet is recommended for:
- Heavy fog or haze conditions
- Night scenes with fog or haze
- Non-homogeneous haze (varies across the image)
- Images where detail preservation is critical
- Higher quality output where processing time is less important

## Model Comparison

| Feature | AOD-Net | Enhanced | DeepDehazeNet |
|---------|---------|----------|---------------|
| Speed | Fast | Medium | Medium-Slow |
| Heavy Haze | Poor | Good | Excellent |
| Detail Preservation | Fair | Good | Excellent |
| Night Scenes | Poor | Fair | Good |
| Color Accuracy | Fair | Good | Excellent |
| Contrast | Low | Medium | High |

## Usage

DeepDehazeNet is now the default model in the web interface. To specifically select it:

1. Upload your hazy image
2. In the model dropdown, select "DeepDehazeNet" 
3. Click "Process Image"

For the API:
```
POST /upload-image
Form data: 
  - file: <image file>
  - model: deep
```

## Technical Details

DeepDehazeNet uses an encoder-decoder architecture with:
- Skip connections for detail preservation
- Attention mechanisms to focus on hazy regions
- Multi-scale processing with dilated convolutions
- Batch normalization for stable training
- Dedicated transmission map estimation branch
- Color correction branch for natural colors

## Troubleshooting

If DeepDehazeNet processes slowly:
- Try using a smaller image resolution
- For real-time applications, consider using the "Light" model instead
- For batch processing, use DeepDehazeNet for optimal quality

## Model Weights

The model weights are stored in `static/models/weights/deep_net.pth`. These weights are optimized for the best dehazing performance across a wide variety of scenes.
