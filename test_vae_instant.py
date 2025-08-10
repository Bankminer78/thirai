#!/usr/bin/env python3
"""
Test VAE processing for instant performance after warmup
"""

import os
import time
import numpy as np
from PIL import Image
import coremltools as ct

def test_instant_processing():
    """Test the concept of keeping models loaded for instant processing"""
    
    print("🔍 Testing VAE instant processing concept...")
    
    # Paths
    size = 256
    root = os.path.dirname(os.path.abspath(__file__))
    enc_path = os.path.join(root, f"sdxl_vae_encoder_{size}x{size}.mlpackage")
    dec_path = os.path.join(root, f"sdxl_vae_decoder_{size}x{size}.mlpackage")
    
    if not os.path.exists(enc_path):
        print(f"❌ Encoder not found: {enc_path}")
        return
        
    if not os.path.exists(dec_path):
        print(f"❌ Decoder not found: {dec_path}")
        return
    
    print("📍 Loading models once (this is the 'cold start')...")
    
    # Load models (this is the one-time cost)
    start_load = time.time()
    enc_model = ct.models.MLModel(enc_path, compute_units=ct.ComputeUnit.ALL)
    dec_model = ct.models.MLModel(dec_path, compute_units=ct.ComputeUnit.ALL)
    load_time = (time.time() - start_load) * 1000
    
    print(f"✅ Models loaded in {load_time:.1f}ms")
    
    # Create test image
    test_image = np.random.rand(size, size, 3).astype(np.float32)
    test_image_pil = Image.fromarray((test_image * 255).astype(np.uint8))
    
    # Prepare input
    img_input = np.transpose(test_image, (2, 0, 1))[None, ...] * 2.0 - 1.0
    
    print("🔥 Warming up models (2x each)...")
    
    # Warmup encoder
    for i in range(2):
        _ = enc_model.predict({"x": img_input})
    
    # Get latent for decoder warmup
    enc_out = enc_model.predict({"x": img_input})
    latent = next(iter(enc_out.values()))
    latent = np.array(latent)
    if latent.shape[1] != 4:
        latent = np.transpose(latent, (0, 3, 1, 2))
    
    # Warmup decoder
    for i in range(2):
        _ = dec_model.predict({"z_scaled": latent})
    
    print("🎉 Models warmed up! Now testing instant processing...")
    
    # Test multiple rounds to show consistency
    for round_num in range(3):
        print(f"\n🚀 Round {round_num + 1}:")
        
        # Encode
        start_enc = time.time()
        enc_out = enc_model.predict({"x": img_input})
        enc_ms = (time.time() - start_enc) * 1000
        
        # Get latent
        latent = next(iter(enc_out.values()))
        latent = np.array(latent)
        if latent.shape[1] != 4:
            latent = np.transpose(latent, (0, 3, 1, 2))
        
        # Decode
        start_dec = time.time()
        dec_out = dec_model.predict({"z_scaled": latent})
        dec_ms = (time.time() - start_dec) * 1000
        
        total_ms = enc_ms + dec_ms
        
        print(f"   ⚡ Encode: {enc_ms:.1f}ms | Decode: {dec_ms:.1f}ms | Total: {total_ms:.1f}ms")
    
    print(f"\n✅ Concept proven! Models stay loaded for instant processing")
    print(f"💡 Cold start cost: {load_time:.1f}ms (one time)")
    print(f"🏃 Hot processing: ~{total_ms:.1f}ms per image")

if __name__ == "__main__":
    test_instant_processing()