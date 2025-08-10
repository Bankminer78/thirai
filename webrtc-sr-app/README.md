# WebRTC Super Resolution App

Real-time video streaming with AI-powered Super Resolution using VAE encoding/decoding.

## Files Included

### Core Application Files
- `conf.html` - Main WebRTC video chat interface with Super Resolution
- `vae_server.py` - FastAPI server for VAE encoding/decoding
- `requirements_server.txt` - Python dependencies

### AI Models
- `sdxl_vae_encoder_256x256.mlpackage` - CoreML VAE encoder (256x256)
- `sdxl_vae_decoder_256x256.mlpackage` - CoreML VAE decoder (256x256)

## Setup Instructions

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements_server.txt
   ```

2. **Start the VAE server:**
   ```bash
   python vae_server.py
   ```
   Server runs on http://127.0.0.1:8000

3. **Open the WebRTC app:**
   Open `conf.html` in a web browser (Chrome/Safari recommended)

## Usage

1. **Choose Role:**
   - Click "SR Sender" to send Super Resolution data
   - Click "SR Receiver" to receive and display SR patches

2. **Super Resolution Controls:**
   - **Sender:** Toggle SR on/off, see green patch indicator
   - **Receiver:** Adjust patch size (50-400px) and image scale (0.5x-3.0x)

3. **Video Chat:**
   - Standard WebRTC peer-to-peer video calling
   - Super Resolution patches sent via data channels
   - Real-time VAE encoding/decoding at ~3 FPS

## Architecture

- **WebRTC:** P2P video streaming + data channels for latent vectors
- **VAE Pipeline:** 256x256 patches encoded → latent → decoded → composited
- **CoreML:** Hardware-accelerated AI processing on macOS
- **Real-time:** ~125ms per patch (encode + network + decode)

## Requirements

- **macOS** (for CoreML models)
- **Python 3.8+**
- **Modern browser** with WebRTC support
- **Camera access** for video streaming