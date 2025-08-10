#!/usr/bin/env python3

import torch
import numpy as np
from diffusers import AutoencoderKL
from PIL import Image
from torchvision import transforms
import argparse
import os

def create_dummy_latent(output_path="latent_fp16.bin", size=48, channels=4):
    """Create a dummy latent tensor and save as FP16 binary"""
    print(f"Creating dummy latent: [{1}, {channels}, {size}, {size}]")
    
    # Create random latent in typical range for SDXL VAE
    latent = torch.randn(1, channels, size, size) * 0.18215  # SDXL VAE scaling factor
    
    # Convert to FP16 and save
    u16 = latent.half().contiguous().cpu().numpy().view("uint16")
    with open(output_path, "wb") as f:
        f.write(u16.tobytes())
    
    print(f"Wrote {output_path} ({u16.size} elements)")
    return output_path

def encode_image_to_latent(image_path, output_path="latent_fp16.bin", model_id="stabilityai/sdxl-vae"):
    """Encode an image to latent using SDXL VAE"""
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return create_dummy_latent(output_path)
    
    try:
        print(f"Loading VAE model: {model_id}")
        vae = AutoencoderKL.from_pretrained(model_id).eval()
        scale = vae.config.scaling_factor
        
        print(f"Loading image: {image_path}")
        img = Image.open(image_path).convert("RGB")
        
        # Resize to 384x384 if needed
        if img.size != (384, 384):
            print(f"Resizing image from {img.size} to (384, 384)")
            img = img.resize((384, 384), Image.Resampling.LANCZOS)
        
        # Convert to tensor
        tfm = transforms.ToTensor()
        x01 = tfm(img).unsqueeze(0)  # [1,3,384,384] in [0,1]
        x = x01 * 2 - 1  # Convert to [-1,1]
        
        print("Encoding image...")
        with torch.no_grad():
            z = vae.encode(x).latent_dist.mean * scale  # [1,4,48,48] FP32
        
        print(f"Latent shape: {list(z.shape)}")
        
        # Convert to FP16 and save
        u16 = z.half().contiguous().cpu().numpy().view("uint16")
        with open(output_path, "wb") as f:
            f.write(u16.tobytes())
        
        print(f"Wrote {output_path} ({u16.size} elements)")
        return output_path
        
    except Exception as e:
        print(f"Error encoding image: {e}")
        print("Falling back to dummy latent...")
        return create_dummy_latent(output_path)

def main():
    parser = argparse.ArgumentParser(description="Create latent binary files for ONNX VAE decoder testing")
    parser.add_argument("--image", "-i", type=str, help="Input image path (optional)")
    parser.add_argument("--output", "-o", type=str, default="latent_fp16.bin", help="Output binary file path")
    parser.add_argument("--dummy", "-d", action="store_true", help="Create dummy latent instead of encoding image")
    parser.add_argument("--size", "-s", type=int, default=48, help="Latent spatial size (default: 48 for 384px image)")
    
    args = parser.parse_args()
    
    if args.dummy or not args.image:
        create_dummy_latent(args.output, args.size)
    else:
        encode_image_to_latent(args.image, args.output)

if __name__ == "__main__":
    main()