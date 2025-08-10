#!/usr/bin/env python3
"""
Demo: Test video frame processing concept
Shows how fast we can process video frames with warm models
"""

import time
import numpy as np
from PIL import Image
import io
import requests

def test_video_frame_processing():
    """Test processing synthetic video frames at different rates"""
    
    print("🎥 Testing VAE video frame processing concept...")
    
    # Create synthetic video frames (simulating webcam input)
    frames = []
    for i in range(10):
        # Random 256x256 RGB frame
        frame = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        frames.append(Image.fromarray(frame))
    
    print(f"📸 Generated {len(frames)} test frames")
    
    # Test different processing rates
    target_fps_rates = [1, 2, 5, 8, 10, 15]
    
    for target_fps in target_fps_rates:
        print(f"\n🎯 Testing {target_fps} FPS processing...")
        
        frame_interval = 1.0 / target_fps
        total_processing_time = 0
        successful_frames = 0
        
        for i, frame in enumerate(frames):
            frame_start = time.time()
            
            try:
                # Convert frame to JPEG blob (simulating browser capture)
                img_buffer = io.BytesIO()
                frame.save(img_buffer, format='JPEG', quality=60)
                img_buffer.seek(0)
                
                # Simulate sending to server
                processing_start = time.time()
                
                # In real implementation, this would be:
                # response = requests.post('http://127.0.0.1:8000/process', 
                #                         files={'file': ('frame.jpg', img_buffer, 'image/jpeg')})
                
                # For demo, simulate the ~125ms processing time we measured
                simulated_processing_time = 0.125  # 125ms
                time.sleep(simulated_processing_time)
                
                processing_end = time.time()
                processing_time = processing_end - processing_start
                total_processing_time += processing_time
                successful_frames += 1
                
                # Check if we can maintain target FPS
                frame_end = time.time()
                frame_total_time = frame_end - frame_start
                
                if frame_total_time > frame_interval:
                    print(f"   ⚠️  Frame {i+1}: {processing_time*1000:.1f}ms (too slow for {target_fps} FPS)")
                else:
                    print(f"   ✅ Frame {i+1}: {processing_time*1000:.1f}ms (on time)")
                
                # Wait for next frame (if we have time left)
                remaining_time = frame_interval - frame_total_time
                if remaining_time > 0:
                    time.sleep(remaining_time)
                    
            except Exception as e:
                print(f"   ❌ Frame {i+1} failed: {e}")
        
        # Calculate stats
        avg_processing_time = total_processing_time / successful_frames if successful_frames > 0 else 0
        theoretical_max_fps = 1.0 / avg_processing_time if avg_processing_time > 0 else 0
        
        print(f"   📊 Results:")
        print(f"      • Avg processing time: {avg_processing_time*1000:.1f}ms")
        print(f"      • Theoretical max FPS: {theoretical_max_fps:.1f}")
        print(f"      • Target {target_fps} FPS: {'✅ ACHIEVABLE' if theoretical_max_fps >= target_fps else '❌ TOO SLOW'}")

def estimate_video_performance():
    """Estimate real-world video processing performance"""
    
    print("\n🔥 VAE Video Processing Performance Estimates:")
    print("=" * 50)
    
    # Based on our measured performance
    encode_time = 67  # ms
    decode_time = 55  # ms  
    total_vae_time = encode_time + decode_time  # ~122ms
    
    # Add overhead estimates
    frame_capture_time = 5    # ms (browser canvas capture)
    network_time = 10         # ms (local network round-trip) 
    display_time = 5          # ms (canvas drawing)
    
    total_time = total_vae_time + frame_capture_time + network_time + display_time
    max_fps = 1000 / total_time
    
    print(f"⚡ VAE Processing Breakdown:")
    print(f"   • Encode: {encode_time}ms")
    print(f"   • Decode: {decode_time}ms")
    print(f"   • Frame capture: {frame_capture_time}ms")
    print(f"   • Network: {network_time}ms")
    print(f"   • Display: {display_time}ms")
    print(f"   • TOTAL: {total_time}ms")
    print(f"")
    print(f"📈 Theoretical Performance:")
    print(f"   • Max sustainable FPS: {max_fps:.1f}")
    print(f"   • Recommended target: {max_fps * 0.8:.1f} FPS (with 20% headroom)")
    print(f"")
    print(f"🎯 Realistic Video Settings:")
    for fps in [5, 8, 10, 12, 15]:
        required_time = 1000 / fps
        if required_time >= total_time:
            status = "✅ SMOOTH"
            headroom = ((required_time - total_time) / required_time) * 100
            print(f"   • {fps} FPS: {status} ({headroom:.1f}% headroom)")
        else:
            status = "❌ CHOPPY"
            overrun = ((total_time - required_time) / required_time) * 100
            print(f"   • {fps} FPS: {status} ({overrun:.1f}% overrun)")

if __name__ == "__main__":
    test_video_frame_processing()
    estimate_video_performance()