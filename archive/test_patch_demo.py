#!/usr/bin/env python3
"""
Test the face patch processing endpoint
"""

import requests
import time
from PIL import Image, ImageDraw
import io

def create_test_image():
    """Create a test image with a face-like region in the center"""
    # Create 640x480 image 
    img = Image.new('RGB', (640, 480), color='lightblue')
    draw = ImageDraw.Draw(img)
    
    # Draw a simple "face" in the center
    center_x, center_y = 320, 240
    
    # Face circle (skin tone)
    draw.ellipse([center_x-80, center_y-80, center_x+80, center_y+80], fill='peachpuff', outline='black', width=2)
    
    # Eyes
    draw.ellipse([center_x-30, center_y-20, center_x-10, center_y], fill='black')
    draw.ellipse([center_x+10, center_y-20, center_x+30, center_y], fill='black')
    
    # Nose
    draw.polygon([center_x-5, center_y+5, center_x+5, center_y+5, center_x, center_y+20], fill='pink')
    
    # Mouth
    draw.arc([center_x-20, center_y+15, center_x+20, center_y+35], start=0, end=180, fill='red', width=3)
    
    # Add some background details
    draw.rectangle([50, 50, 150, 150], fill='green', outline='darkgreen')
    draw.rectangle([490, 320, 590, 420], fill='red', outline='darkred')
    
    return img

def test_patch_processing():
    """Test the patch processing endpoint"""
    
    print("🎯 Testing Face Patch Processing...")
    
    # Create test image
    test_img = create_test_image()
    test_img.save('test_face.jpg', quality=90)
    print("📸 Created test image: test_face.jpg")
    
    # Convert to bytes for upload
    img_buffer = io.BytesIO()
    test_img.save(img_buffer, format='JPEG', quality=90)
    img_buffer.seek(0)
    
    try:
        # Test the patch processing endpoint
        print("🚀 Sending image to patch processing endpoint...")
        
        start_time = time.time()
        response = requests.post(
            'http://127.0.0.1:8000/process_patch',
            files={'file': ('test_face.jpg', img_buffer, 'image/jpeg')},
            timeout=30
        )
        end_time = time.time()
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Patch processing successful!")
            print(f"⏱️  Total time: {(end_time - start_time)*1000:.1f}ms")
            print(f"🔧 VAE Encode: {result['timing']['encode_ms']}ms")
            print(f"🔧 VAE Decode: {result['timing']['decode_ms']}ms")
            print(f"📊 Patch coordinates: {result['patch_coords']}")
            print(f"📈 Latent stats: shape={result['latent_stats']['shape']}, mean={result['latent_stats']['mean']:.3f}")
            
            # Save results
            import base64
            
            # Save composited result
            composited_data = base64.b64decode(result['composited_image_b64'])
            with open('test_patch_composited.jpg', 'wb') as f:
                f.write(composited_data)
            print("💾 Saved composited result: test_patch_composited.jpg")
            
            # Save processed patch
            patch_data = base64.b64decode(result['processed_patch_b64'])
            with open('test_processed_patch.jpg', 'wb') as f:
                f.write(patch_data)
            print("💾 Saved processed patch: test_processed_patch.jpg")
            
            print("\n🎉 Face Patch Demo is ready!")
            print("🌐 Open http://127.0.0.1:8000 and click 'Face Patch Demo' tab")
            
        else:
            print(f"❌ Request failed: {response.status_code}")
            print(f"📄 Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to server. Make sure the server is running!")
        print("💡 Start it with: python vae_server.py")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_patch_processing()