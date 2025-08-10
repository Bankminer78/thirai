#!/usr/bin/env python3
"""
FastAPI VAE Encoding/Decoding Server
Keeps models loaded in memory for instant processing (no cold starts)
"""

import os
import time
import json
import logging
import numpy as np
from PIL import Image
import io
import base64
from typing import Optional

import coremltools as ct
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# PyTorch imports for custom AE
try:
    import torch
    import torch.nn as nn
    from torchvision import transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if TORCH_AVAILABLE:
    logger.info("✅ PyTorch available for custom AE")
else:
    logger.warning("⚠️ PyTorch not available - custom AE disabled")

class VAEProcessor:
    def __init__(self, size: int = 256):
        self.size = size
        self.root = os.path.dirname(os.path.abspath(__file__))
        self.enc_path = os.path.join(self.root, f"sdxl_vae_encoder_{size}x{size}.mlpackage")
        self.dec_path = os.path.join(self.root, f"sdxl_vae_decoder_{size}x{size}.mlpackage")
        
        logger.info(f"🔍 Initializing VAE Processor (size={size})")
        logger.info(f"📁 Encoder path: {self.enc_path}")
        logger.info(f"📁 Decoder path: {self.dec_path}")
        
        # Validate paths
        if not os.path.exists(self.enc_path):
            raise FileNotFoundError(f"Encoder model not found: {self.enc_path}")
        if not os.path.exists(self.dec_path):
            raise FileNotFoundError(f"Decoder model not found: {self.dec_path}")
            
        # Load models
        logger.info("⚙️  Loading encoder model...")
        start_time = time.time()
        self.enc_model = ct.models.MLModel(self.enc_path, compute_units=ct.ComputeUnit.ALL)
        enc_load_time = (time.time() - start_time) * 1000
        logger.info(f"✅ Encoder loaded in {enc_load_time:.1f}ms")
        
        logger.info("⚙️  Loading decoder model...")
        start_time = time.time()
        self.dec_model = ct.models.MLModel(self.dec_path, compute_units=ct.ComputeUnit.ALL)
        dec_load_time = (time.time() - start_time) * 1000
        logger.info(f"✅ Decoder loaded in {dec_load_time:.1f}ms")
        
        # Warm up models with dummy data
        logger.info("🔥 Warming up models...")
        self._warmup()
        logger.info("🎉 VAE Processor ready for instant processing!")
    
    def _warmup(self):
        """Warm up both models to avoid cold starts"""
        dummy_input = np.random.randn(1, 3, self.size, self.size).astype(np.float32) * 2 - 1
        
        # Warmup encoder (2x for good measure)
        for i in range(2):
            _ = self.enc_model.predict({"x": dummy_input})
        
        # Get dummy latent for decoder warmup
        enc_out = self.enc_model.predict({"x": dummy_input})
        dummy_latent = next(iter(enc_out.values()))
        dummy_latent = np.array(dummy_latent)
        
        # Ensure correct shape (NCHW)
        if dummy_latent.shape[1] != 4:
            dummy_latent = np.transpose(dummy_latent, (0, 3, 1, 2))
        
        # Warmup decoder (2x for good measure)
        for i in range(2):
            _ = self.dec_model.predict({"z_scaled": dummy_latent})
    
    def preprocess_image(self, pil_image: Image.Image) -> np.ndarray:
        """Convert PIL image to model input format [-1, 1] NCHW"""
        # Resize to target size
        img = pil_image.convert("RGB").resize((self.size, self.size), Image.BICUBIC)
        
        # Convert to numpy [0, 1]
        img_array = np.array(img).astype(np.float32) / 255.0
        
        # Convert to NCHW format
        img_nchw = np.transpose(img_array, (2, 0, 1))[None, ...]
        
        # Scale to [-1, 1]
        img_input = img_nchw * 2.0 - 1.0
        
        return img_input
    
    def postprocess_output(self, output: np.ndarray) -> np.ndarray:
        """Convert model output to [0, 1] NCHW format"""
        # Ensure NCHW format
        if output.shape[1] != 3:
            output = np.transpose(output, (0, 3, 1, 2))
        
        # Convert from [-1, 1] to [0, 1]
        output_01 = np.clip((output + 1) / 2, 0, 1)
        
        return output_01
    
    def encode_decode(self, pil_image: Image.Image) -> dict:
        """Perform full encode-decode roundtrip"""
        start_total = time.time()
        
        # Preprocess
        input_array = self.preprocess_image(pil_image)
        
        # Encode
        start_enc = time.time()
        enc_out = self.enc_model.predict({"x": input_array})
        enc_ms = (time.time() - start_enc) * 1000
        
        # Extract latent
        latent = next(iter(enc_out.values()))
        latent = np.array(latent)
        
        # Ensure NCHW format for decoder
        if latent.shape[1] != 4:
            latent = np.transpose(latent, (0, 3, 1, 2))
        
        # Decode
        start_dec = time.time()
        dec_out = self.dec_model.predict({"z_scaled": latent})
        dec_ms = (time.time() - start_dec) * 1000
        
        # Extract and postprocess output
        output = next(iter(dec_out.values()))
        output = np.array(output)
        output_01 = self.postprocess_output(output)
        
        total_ms = (time.time() - start_total) * 1000
        
        # Convert back to PIL Image
        output_hwc = (output_01[0].transpose(1, 2, 0) * 255).astype(np.uint8)
        output_pil = Image.fromarray(output_hwc)
        
        # Compute latent stats
        latent_stats = {
            "shape": list(latent.shape),
            "min": float(latent.min()),
            "max": float(latent.max()),
            "mean": float(latent.mean()),
            "std": float(latent.std())
        }
        
        return {
            "output_image": output_pil,
            "latent_stats": latent_stats,
            "timing": {
                "encode_ms": round(enc_ms, 2),
                "decode_ms": round(dec_ms, 2),
                "total_ms": round(total_ms, 2)
            }
        }

# PyTorch Custom Autoencoder Classes (from classic_ae/inference_metal.py)
if TORCH_AVAILABLE:
    class Block(nn.Module):
        def __init__(self, in_ch, out_ch, group_norm_groups=8):
            super().__init__()
            num_groups = min(group_norm_groups, out_ch // 4 if out_ch > 4 else out_ch)
            if num_groups == 0: num_groups = 1
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.GroupNorm(num_groups, out_ch),
                nn.SiLU(),
            )
        def forward(self, x):
            return self.conv(x)

    class Encoder(nn.Module):
        def __init__(self, in_ch=3, model_ch=64, lat_ch=128):
            super().__init__()
            self.inc = Block(in_ch, model_ch)
            self.down1 = nn.Sequential(nn.MaxPool2d(2), Block(model_ch, model_ch * 2))
            self.down2 = nn.Sequential(nn.MaxPool2d(2), Block(model_ch * 2, model_ch * 4))
            self.bot = nn.Conv2d(model_ch * 4, lat_ch, 1)

        def forward(self, x):
            x1 = self.inc(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            latent = self.bot(x3)
            return latent, x1, x2, x3

    class Decoder(nn.Module):
        def __init__(self, out_ch=3, model_ch=64, lat_ch=128):
            super().__init__()
            self.bot = nn.Conv2d(lat_ch, model_ch * 4, 1)
            self.up1 = nn.Sequential(Block(model_ch * 8, model_ch * 2), nn.Upsample(scale_factor=2, mode='bilinear'))
            self.up2 = nn.Sequential(Block(model_ch * 4, model_ch), nn.Upsample(scale_factor=2, mode='bilinear'))
            self.outc = nn.Conv2d(model_ch * 2, out_ch, 1)

        def forward(self, latent, x1, x2, x3):
            b = self.bot(latent)
            up1 = self.up1(torch.cat([b, x3], dim=1))
            up2 = self.up2(torch.cat([up1, x2], dim=1))
            logits = self.outc(torch.cat([up2, x1], dim=1))
            return torch.sigmoid(logits)

    class CustomAEProcessor:
        def __init__(self):
            # Configuration from classic_ae
            self.ROI = 256
            self.LATENT_CH = 128  
            self.MODEL_CH = 64
            
            # Setup device (Metal MPS if available)
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
                logger.info("🚀 Using Metal Performance Shaders (MPS) device")
            else:
                self.device = torch.device("cpu")
                logger.info("🔄 Using CPU device")
            
            self.root = os.path.dirname(os.path.abspath(__file__))
            encoder_path = os.path.join(self.root, "encoder_best.pt")
            decoder_path = os.path.join(self.root, "decoder_best.pt")
            
            # Validate model files exist
            if not os.path.exists(encoder_path):
                raise FileNotFoundError(f"Encoder model not found: {encoder_path}")
            if not os.path.exists(decoder_path):
                raise FileNotFoundError(f"Decoder model not found: {decoder_path}")
                
            # Initialize models
            logger.info("⚙️  Loading custom PyTorch encoder...")
            self.encoder = Encoder(in_ch=3, model_ch=self.MODEL_CH, lat_ch=self.LATENT_CH).to(self.device)
            self.encoder.load_state_dict(torch.load(encoder_path, map_location=self.device))
            self.encoder.eval()
            
            logger.info("⚙️  Loading custom PyTorch decoder...")
            self.decoder = Decoder(out_ch=3, model_ch=self.MODEL_CH, lat_ch=self.LATENT_CH).to(self.device)
            self.decoder.load_state_dict(torch.load(decoder_path, map_location=self.device))
            self.decoder.eval()
            
            # Setup transforms
            self.transform = transforms.Compose([
                transforms.Resize((self.ROI, self.ROI)),
                transforms.ToTensor(),
            ])
            self.to_pil = transforms.ToPILImage()
            
            # Warmup
            logger.info("🔥 Warming up custom AE models...")
            self._warmup()
            logger.info("✅ Custom AE Processor ready!")
            
        def _warmup(self):
            """Warm up both models to avoid cold starts"""
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, self.ROI, self.ROI).to(self.device)
                
                # Warmup encoder
                latent, x1, x2, x3 = self.encoder(dummy_input)
                
                # Warmup decoder  
                _ = self.decoder(latent, x1, x2, x3)
                
                # Synchronize if using MPS
                if self.device.type == "mps":
                    torch.mps.synchronize()
                    
        def encode_decode(self, pil_image: Image.Image) -> dict:
            """Perform full encode-decode roundtrip"""
            start_total = time.time()
            
            # Preprocess image
            image_tensor = self.transform(pil_image.convert("RGB")).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                # Encode
                start_enc = time.time()
                latent, x1, x2, x3 = self.encoder(image_tensor)
                if self.device.type == "mps":
                    torch.mps.synchronize()
                enc_ms = (time.time() - start_enc) * 1000
                
                # Decode
                start_dec = time.time()
                reconstructed_tensor = self.decoder(latent, x1, x2, x3)
                if self.device.type == "mps":
                    torch.mps.synchronize()
                dec_ms = (time.time() - start_dec) * 1000
                
            total_ms = (time.time() - start_total) * 1000
            
            # Convert back to PIL Image
            output_pil = self.to_pil(reconstructed_tensor.squeeze().cpu())
            
            # Compute latent stats
            latent_np = latent.cpu().numpy()
            latent_stats = {
                "shape": list(latent_np.shape),
                "min": float(latent_np.min()),
                "max": float(latent_np.max()),
                "mean": float(latent_np.mean()),
                "std": float(latent_np.std())
            }
            
            return {
                "output_image": output_pil,
                "latent_stats": latent_stats,
                "timing": {
                    "encode_ms": round(enc_ms, 2),
                    "decode_ms": round(dec_ms, 2),
                    "total_ms": round(total_ms, 2)
                }
            }

# Initialize processors globally (loaded once at startup)
logger.info("🚀 Initializing processors...")

# Initialize CoreML VAE Processor
try:
    processor = VAEProcessor(size=256)
    logger.info("✅ CoreML VAE Processor initialized successfully!")
except Exception as e:
    logger.error(f"❌ Failed to initialize CoreML VAE Processor: {e}")
    processor = None

# Initialize Custom AE Processor (PyTorch)
custom_processor = None
if TORCH_AVAILABLE:
    try:
        custom_processor = CustomAEProcessor()
        logger.info("✅ Custom AE Processor initialized successfully!")
    except Exception as e:
        logger.error(f"❌ Failed to initialize Custom AE Processor: {e}")
        custom_processor = None

# WebSocket Connection Manager
class SignalingManager:
    def __init__(self):
        self.connections = {}
        self.rooms = {}
    
    async def connect(self, websocket: WebSocket, user_id: str, room: str):
        await websocket.accept()
        self.connections[user_id] = websocket
        
        if room not in self.rooms:
            self.rooms[room] = set()
        self.rooms[room].add(user_id)
        
        logger.info(f"👥 User {user_id} joined room {room}")
        
        # Notify other users in room
        await self.broadcast_to_room(room, {
            "type": "user_joined",
            "user_id": user_id
        }, exclude=user_id)
    
    def disconnect(self, user_id: str):
        if user_id in self.connections:
            # Find which room the user was in
            for room, users in self.rooms.items():
                if user_id in users:
                    users.remove(user_id)
                    # Notify other users
                    self.broadcast_to_room_sync(room, {
                        "type": "user_left", 
                        "user_id": user_id
                    })
                    break
            del self.connections[user_id]
            logger.info(f"👤 User {user_id} disconnected")
    
    async def send_to_user(self, user_id: str, message: dict):
        if user_id in self.connections:
            try:
                await self.connections[user_id].send_text(json.dumps(message))
            except:
                self.disconnect(user_id)
    
    async def broadcast_to_room(self, room: str, message: dict, exclude: str = None):
        if room in self.rooms:
            for user_id in self.rooms[room].copy():
                if user_id != exclude:
                    await self.send_to_user(user_id, message)
    
    def broadcast_to_room_sync(self, room: str, message: dict):
        # Synchronous version for use in disconnect
        import asyncio
        if room in self.rooms:
            for user_id in self.rooms[room].copy():
                try:
                    asyncio.create_task(self.send_to_user(user_id, message))
                except:
                    pass

signaling_manager = SignalingManager()

# FastAPI app
app = FastAPI(title="VAE Encoding/Decoding Server", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
async def get_web_interface():
    """Web interface with both image and live video processing"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>VAE Live Video Processor</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 1400px; margin: 0 auto; padding: 20px; background: #1a1a1a; color: white; }
            .tabs { display: flex; background: #2a2a2a; border-radius: 8px; margin-bottom: 20px; overflow: hidden; }
            .tab { flex: 1; padding: 15px; text-align: center; cursor: pointer; transition: all 0.3s; }
            .tab:hover { background: #3a3a3a; }
            .tab.active { background: #007bff; color: white; }
            .tab-content { display: none; }
            .tab-content.active { display: block; }
            
            .video-container { display: flex; gap: 20px; margin: 20px 0; }
            .video-box { flex: 1; background: #2a2a2a; border-radius: 10px; padding: 20px; text-align: center; }
            .video-box h3 { margin: 0 0 15px 0; color: #00ff88; }
            video, canvas { max-width: 100%; height: auto; border-radius: 8px; background: #000; }
            
            .controls { background: #2a2a2a; padding: 20px; border-radius: 10px; margin: 20px 0; }
            .control-group { display: flex; align-items: center; gap: 15px; margin: 15px 0; flex-wrap: wrap; }
            .control-group label { min-width: 120px; color: #ccc; }
            
            button { background: #007bff; color: white; border: none; padding: 12px 24px; border-radius: 6px; cursor: pointer; font-size: 14px; transition: all 0.3s; }
            button:hover { background: #0056b3; transform: translateY(-1px); }
            button:disabled { background: #666; cursor: not-allowed; transform: none; }
            button.danger { background: #dc3545; }
            button.danger:hover { background: #c82333; }
            button.success { background: #28a745; }
            button.success:hover { background: #218838; }
            
            input[type="range"] { flex: 1; min-width: 200px; }
            
            .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }
            .stat-box { background: #2a2a2a; padding: 15px; border-radius: 8px; border-left: 4px solid #007bff; }
            .stat-value { font-size: 24px; font-weight: bold; color: #00ff88; }
            .stat-label { color: #ccc; font-size: 12px; text-transform: uppercase; }
            
            .performance { background: #1a3a1a; padding: 15px; border-radius: 8px; font-family: monospace; border: 1px solid #28a745; }
            
            /* Image tab styles */
            .upload-area { border: 2px dashed #555; padding: 40px; text-align: center; cursor: pointer; border-radius: 10px; transition: all 0.3s; }
            .upload-area:hover { background-color: #2a2a2a; border-color: #007bff; }
            .image-container { flex: 1; }
            .image-container img { max-width: 100%; height: auto; border-radius: 8px; }
            
            .fps-indicator { 
                position: absolute; top: 10px; right: 10px; 
                background: rgba(0,0,0,0.7); padding: 5px 10px; 
                border-radius: 5px; font-size: 12px; color: #00ff88;
                font-family: monospace;
            }
            .video-box { position: relative; }
            
            /* Face Patch Demo styles */
            .patch-overlay { 
                position: absolute; 
                border: 3px solid #00ff88; 
                border-radius: 8px;
                box-shadow: 0 0 20px rgba(0, 255, 136, 0.6);
                pointer-events: none;
                animation: pulse-border 2s infinite;
            }
            
            @keyframes pulse-border {
                0% { border-color: #00ff88; box-shadow: 0 0 20px rgba(0, 255, 136, 0.6); }
                50% { border-color: #00aaff; box-shadow: 0 0 30px rgba(0, 170, 255, 0.8); }
                100% { border-color: #00ff88; box-shadow: 0 0 20px rgba(0, 255, 136, 0.6); }
            }
            
            .processing-indicator {
                position: absolute;
                top: 50%; left: 50%;
                transform: translate(-50%, -50%);
                background: rgba(0, 255, 136, 0.9);
                color: black;
                padding: 8px 16px;
                border-radius: 20px;
                font-size: 12px;
                font-weight: bold;
                z-index: 100;
                animation: processing-pulse 1s infinite;
            }
            
            @keyframes processing-pulse {
                0% { transform: translate(-50%, -50%) scale(1); }
                50% { transform: translate(-50%, -50%) scale(1.1); }
                100% { transform: translate(-50%, -50%) scale(1); }
            }
            
            .demo-container {
                display: flex;
                gap: 20px;
                align-items: flex-start;
            }
            
            .demo-main { flex: 2; }
            .demo-sidebar { flex: 1; }
            
            .patch-preview {
                background: #2a2a2a;
                padding: 15px;
                border-radius: 10px;
                margin-bottom: 20px;
                text-align: center;
            }
            
            .patch-preview img {
                max-width: 100%;
                border-radius: 8px;
                border: 2px solid #007bff;
            }
            
            .demo-info {
                background: linear-gradient(135deg, #1a3a1a, #2a2a2a);
                padding: 20px;
                border-radius: 10px;
                border: 1px solid #00ff88;
            }
            
            .demo-step {
                display: flex;
                align-items: center;
                margin: 10px 0;
                padding: 10px;
                background: rgba(0, 255, 136, 0.1);
                border-radius: 5px;
                border-left: 3px solid #00ff88;
            }
            
            .step-icon {
                font-size: 20px;
                margin-right: 10px;
                min-width: 30px;
            }
        </style>
    </head>
    <body>
        <h1>🎥🔥 VAE Live Video Processor</h1>
        <p>Real-time video processing with instant VAE encoding/decoding</p>
        
        <div class="tabs">
            <div class="tab active" onclick="switchTab('video')">🎥 Live Video</div>
            <div class="tab" onclick="switchTab('patch')">🎯 Face Patch Demo</div>
            <div class="tab" onclick="switchTab('image')">🖼️ Image Upload</div>
        </div>
        
        <!-- Live Video Tab -->
        <div id="video-tab" class="tab-content active">
            <div class="controls">
                <div class="control-group">
                    <button id="startBtn" onclick="startVideo()">📹 Start Camera</button>
                    <button id="stopBtn" onclick="stopVideo()" disabled>⏹️ Stop</button>
                    <button id="processToggle" onclick="toggleProcessing()" disabled>🚀 Start Processing</button>
                </div>
                
                <div class="control-group">
                    <label>Processing FPS:</label>
                    <input type="range" id="fpsSlider" min="1" max="30" value="10" oninput="updateFPS(this.value)">
                    <span id="fpsValue">10</span> FPS
                </div>
                
                <div class="control-group">
                    <label>🔥 Processor:</label>
                    <input type="radio" id="customVideoRadio" name="videoProcessor" value="custom" checked>
                    <label for="customVideoRadio">Custom AE (PyTorch)</label>
                    <input type="radio" id="coremlVideoRadio" name="videoProcessor" value="coreml">
                    <label for="coremlVideoRadio">CoreML VAE</label>
                </div>
                
                <div class="control-group">
                    <label>Quality:</label>
                    <select id="qualitySelect" onchange="updateQuality()">
                        <option value="0.8">High (0.8)</option>
                        <option value="0.6" selected>Medium (0.6)</option>
                        <option value="0.4">Low (0.4)</option>
                    </select>
                </div>
            </div>
            
            <div class="video-container">
                <div class="video-box">
                    <h3>📹 Live Camera</h3>
                    <video id="videoElement" autoplay muted></video>
                    <div class="fps-indicator" id="inputFPS">0 FPS</div>
                </div>
                <div class="video-box">
                    <h3>🎨 Processed VAE</h3>
                    <canvas id="outputCanvas"></canvas>
                    <div class="fps-indicator" id="outputFPS">0 FPS</div>
                </div>
            </div>
            
            <div class="stats-grid">
                <div class="stat-box">
                    <div class="stat-value" id="encodeTime">0</div>
                    <div class="stat-label">Encode Time (ms)</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value" id="decodeTime">0</div>
                    <div class="stat-label">Decode Time (ms)</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value" id="totalTime">0</div>
                    <div class="stat-label">Total Time (ms)</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value" id="framesProcessed">0</div>
                    <div class="stat-label">Frames Processed</div>
                </div>
            </div>
        </div>
        
        <!-- Face Patch Demo Tab -->
        <div id="patch-tab" class="tab-content">
            <div class="demo-container">
                <div class="demo-main">
                    <div class="controls">
                        <div class="control-group">
                            <button id="startPatchBtn" onclick="startPatchDemo()">🎯 Start Face Patch Demo</button>
                            <button id="stopPatchBtn" onclick="stopPatchDemo()" disabled>⏹️ Stop Demo</button>
                            <button id="patchProcessToggle" onclick="togglePatchProcessing()" disabled>🚀 Start VAE Processing</button>
                        </div>
                        
                        <div class="control-group">
                            <label>Processing Rate:</label>
                            <input type="range" id="patchFpsSlider" min="1" max="8" value="3" oninput="updatePatchFPS(this.value)">
                            <span id="patchFpsValue">3</span> FPS
                        </div>
                        
                        <div class="control-group">
                            <label>Patch Size:</label>
                            <select id="patchSizeSelect" onchange="updatePatchSize()">
                                <option value="256" selected>256×256 (Face Focus)</option>
                                <option value="200">200×200 (Tight Crop)</option>
                                <option value="300">300×300 (Wide Crop)</option>
                            </select>
                        </div>
                        
                        <div class="control-group">
                            <label>Bandwidth Simulation:</label>
                            <button id="throttleToggle" onclick="toggleStreamThrottle()">📺 Enable Choppy Stream</button>
                            <br>
                            <label id="choppinessLabel" style="display: none;">Choppiness Level:</label>
                            <input type="range" id="choppinessSlider" min="1" max="10" value="3" oninput="updateChoppiness(this.value)" style="display: none;">
                            <span id="choppinessValue" style="display: none;">Low</span>
                        </div>
                    </div>
                    
                    <div class="video-container">
                        <div class="video-box">
                            <h3>🎯 VAE Processed Output</h3>
                            <canvas id="patchCanvas"></canvas>
                            <div class="processing-indicator" id="processingIndicator" style="display: none;">🔥 VAE Processing...</div>
                            <div class="fps-indicator" id="patchOutputFPS">0 FPS</div>
                        </div>
                    </div>
                    
                    <div class="video-container">
                        <div class="video-box">
                            <h3>📹 Live Camera Feed</h3>
                            <video id="patchVideo" autoplay muted></video>
                            <canvas id="liveCanvas" style="display: none;"></canvas>
                            <div class="patch-overlay" id="patchOverlay" style="display: none;"></div>
                            <div class="fps-indicator" id="patchInputFPS">0 FPS</div>
                        </div>
                    </div>
                    
                    <div class="stats-grid">
                        <div class="stat-box">
                            <div class="stat-value" id="patchEncodeTime">0</div>
                            <div class="stat-label">Encode Time (ms)</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-value" id="patchDecodeTime">0</div>
                            <div class="stat-label">Decode Time (ms)</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-value" id="patchTotalTime">0</div>
                            <div class="stat-label">Total Time (ms)</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-value" id="patchFramesProcessed">0</div>
                            <div class="stat-label">Patches Processed</div>
                        </div>
                    </div>
                </div>
                
                <div class="demo-sidebar">
                    <div class="patch-preview">
                        <h4>🎨 Processed Patch</h4>
                        <img id="patchPreview" src="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' width='256' height='256'><rect width='100%' height='100%' fill='%23333'/><text x='50%' y='50%' text-anchor='middle' fill='%23666' font-family='Arial' font-size='16'>Patch will appear here</text></svg>" alt="Processed patch">
                    </div>
                    
                    <div class="demo-info">
                        <h4>🎯 Face Patch Demo</h4>
                        <div class="demo-step">
                            <div class="step-icon">📹</div>
                            <div>Camera captures your face in real-time</div>
                        </div>
                        <div class="demo-step">
                            <div class="step-icon">🎯</div>
                            <div>Green box shows 256×256 patch region</div>
                        </div>
                        <div class="demo-step">
                            <div class="step-icon">🤖</div>
                            <div>VAE encodes & decodes just the patch</div>
                        </div>
                        <div class="demo-step">
                            <div class="step-icon">✨</div>
                            <div>Processed patch composited back to video</div>
                        </div>
                        <div class="demo-step">
                            <div class="step-icon">⚡</div>
                            <div>Real-time AI processing at ~125ms/patch</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Image Upload Tab -->
        <div id="image-tab" class="tab-content">
            <div class="upload-area" onclick="document.getElementById('fileInput').click()">
                <input type="file" id="fileInput" accept="image/*" style="display: none">
                📁 Click to select image or drag & drop
            </div>
            
            <div class="control-group">
                <label>🔥 Processor:</label>
                <input type="radio" id="customImageRadio" name="imageProcessor" value="custom" checked>
                <label for="customImageRadio">Custom AE (PyTorch)</label>
                <input type="radio" id="coremlImageRadio" name="imageProcessor" value="coreml">
                <label for="coremlImageRadio">CoreML VAE</label>
            </div>
            
            <button id="processBtn" onclick="processImage()" disabled>🚀 Process Image</button>
            
            <div id="results" style="display: none">
                <div class="performance" id="timing"></div>
                <div class="video-container">
                    <div class="image-container">
                        <h3>Original Image</h3>
                        <img id="originalImg" alt="Original">
                    </div>
                    <div class="image-container">
                        <h3>Decoded Image</h3>
                        <img id="decodedImg" alt="Decoded">
                    </div>
                </div>
                <div class="performance">
                    <h3>Latent Statistics</h3>
                    <pre id="stats"></pre>
                </div>
            </div>
        </div>

        <script>
            // Global variables
            let mediaStream = null;
            let isProcessing = false;
            let processingFPS = 10;
            let quality = 0.6;
            let frameCount = 0;
            let processedFrames = 0;
            let lastFrameTime = 0;
            let inputFPSCounter = 0;
            let outputFPSCounter = 0;
            
            // Face patch demo variables
            let patchMediaStream = null;
            let isPatchProcessing = false;
            let patchProcessingFPS = 3;
            let choppinessLevel = 3; // 1-10 scale (1=very choppy, 10=smooth)
            let patchSize = 256;
            let patchFramesProcessed = 0;
            let patchInputFPSCounter = 0;
            let patchOutputFPSCounter = 0;
            let streamRenderInterval = null;
            let streamThrottleEnabled = false;
            
            // Tab switching
            function switchTab(tabName) {
                document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
                document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
                document.querySelector(`[onclick="switchTab('${tabName}')"]`).classList.add('active');
                document.getElementById(`${tabName}-tab`).classList.add('active');
            }
            
            // Video functions
            async function startVideo() {
                try {
                    mediaStream = await navigator.mediaDevices.getUserMedia({ 
                        video: { 
                            width: 640, 
                            height: 480,
                            frameRate: 30
                        } 
                    });
                    
                    const video = document.getElementById('videoElement');
                    video.srcObject = mediaStream;
                    
                    document.getElementById('startBtn').disabled = true;
                    document.getElementById('stopBtn').disabled = false;
                    document.getElementById('processToggle').disabled = false;
                    
                    // Start FPS counters
                    startFPSCounters();
                    
                } catch (error) {
                    alert('Error accessing camera: ' + error.message);
                }
            }
            
            function stopVideo() {
                if (mediaStream) {
                    mediaStream.getTracks().forEach(track => track.stop());
                    mediaStream = null;
                }
                
                if (isProcessing) {
                    toggleProcessing();
                }
                
                document.getElementById('startBtn').disabled = false;
                document.getElementById('stopBtn').disabled = true;
                document.getElementById('processToggle').disabled = true;
                
                // Clear video elements
                document.getElementById('videoElement').srcObject = null;
                const canvas = document.getElementById('outputCanvas');
                const ctx = canvas.getContext('2d');
                ctx.clearRect(0, 0, canvas.width, canvas.height);
            }
            
            function toggleProcessing() {
                isProcessing = !isProcessing;
                const btn = document.getElementById('processToggle');
                
                if (isProcessing) {
                    btn.textContent = '⏸️ Stop Processing';
                    btn.className = 'danger';
                    startProcessingLoop();
                } else {
                    btn.textContent = '🚀 Start Processing';
                    btn.className = '';
                }
            }
            
            function updateFPS(value) {
                processingFPS = parseInt(value);
                document.getElementById('fpsValue').textContent = value;
            }
            
            function updateQuality() {
                quality = parseFloat(document.getElementById('qualitySelect').value);
            }
            
            async function startProcessingLoop() {
                const video = document.getElementById('videoElement');
                const canvas = document.getElementById('outputCanvas');
                const ctx = canvas.getContext('2d');
                
                // Set canvas size to match video
                canvas.width = video.videoWidth || 640;
                canvas.height = video.videoHeight || 480;
                
                const frameInterval = 1000 / processingFPS;
                
                async function processFrame() {
                    if (!isProcessing) return;
                    
                    try {
                        // Capture frame from video
                        const tempCanvas = document.createElement('canvas');
                        tempCanvas.width = 256;  // Match VAE input size
                        tempCanvas.height = 256;
                        const tempCtx = tempCanvas.getContext('2d');
                        tempCtx.drawImage(video, 0, 0, 256, 256);
                        
                        // Convert to blob
                        const blob = await new Promise(resolve => {
                            tempCanvas.toBlob(resolve, 'image/jpeg', quality);
                        });
                        
                        // Send to VAE processing
                        const formData = new FormData();
                        formData.append('file', blob, 'frame.jpg');
                        
                        // Get selected processor
                        const selectedProcessor = document.querySelector('input[name="videoProcessor"]:checked').value;
                        formData.append('processor', selectedProcessor);
                        
                        const startTime = performance.now();
                        const response = await fetch('/process_patch_combined', {
                            method: 'POST',
                            body: formData
                        });
                        
                        if (response.ok) {
                            const result = await response.json();
                            const processTime = performance.now() - startTime;
                            
                            // Update performance stats
                            document.getElementById('encodeTime').textContent = result.timing.encode_ms;
                            document.getElementById('decodeTime').textContent = result.timing.decode_ms;
                            document.getElementById('totalTime').textContent = Math.round(processTime);
                            document.getElementById('framesProcessed').textContent = ++processedFrames;
                            
                            // Display processed frame - handle different response formats
                            const img = new Image();
                            img.onload = () => {
                                ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
                                outputFPSCounter++;
                            };
                            const imageB64 = result.output_image_b64 || result.composited_image_b64;
                            img.src = 'data:image/jpeg;base64,' + imageB64;
                        }
                        
                    } catch (error) {
                        console.error('Frame processing error:', error);
                    }
                    
                    // Schedule next frame
                    if (isProcessing) {
                        setTimeout(processFrame, frameInterval);
                    }
                }
                
                processFrame();
            }
            
            function startFPSCounters() {
                const video = document.getElementById('videoElement');
                
                // Input FPS counter
                function updateInputFPS() {
                    if (video.videoWidth > 0) {
                        inputFPSCounter++;
                    }
                    requestAnimationFrame(updateInputFPS);
                }
                updateInputFPS();
                
                // Update FPS displays every second
                setInterval(() => {
                    document.getElementById('inputFPS').textContent = inputFPSCounter + ' FPS';
                    document.getElementById('outputFPS').textContent = outputFPSCounter + ' FPS';
                    inputFPSCounter = 0;
                    outputFPSCounter = 0;
                }, 1000);
            }
            
            // Face Patch Demo Functions
            async function startPatchDemo() {
                try {
                    patchMediaStream = await navigator.mediaDevices.getUserMedia({ 
                        video: { 
                            width: 640, 
                            height: 480,
                            frameRate: 30
                        } 
                    });
                    
                    const video = document.getElementById('patchVideo');
                    const outputCanvas = document.getElementById('patchCanvas');
                    const liveCanvas = document.getElementById('liveCanvas');
                    
                    video.srcObject = patchMediaStream;
                    
                    // Set up canvases
                    outputCanvas.width = 640;
                    outputCanvas.height = 480;
                    liveCanvas.width = 640;
                    liveCanvas.height = 480;
                    
                    // Wait for video to load, then set up everything
                    video.onloadedmetadata = () => {
                        // Show patch overlay box on live video (not canvas initially)
                        showPatchOverlay(video);
                        startPatchFPSCounters();
                        
                        console.log('Video loaded:', {
                            width: video.videoWidth,
                            height: video.videoHeight,
                            readyState: video.readyState
                        });
                    };
                    
                    document.getElementById('startPatchBtn').disabled = true;
                    document.getElementById('stopPatchBtn').disabled = false;
                    document.getElementById('patchProcessToggle').disabled = false;
                    
                } catch (error) {
                    alert('Error accessing camera: ' + error.message);
                }
            }
            
            function stopPatchDemo() {
                if (patchMediaStream) {
                    patchMediaStream.getTracks().forEach(track => track.stop());
                    patchMediaStream = null;
                }
                
                if (isPatchProcessing) {
                    togglePatchProcessing();
                }
                
                // Stop stream throttling
                stopStreamThrottle();
                
                document.getElementById('startPatchBtn').disabled = false;
                document.getElementById('stopPatchBtn').disabled = true;
                document.getElementById('patchProcessToggle').disabled = true;
                
                // Hide overlay and clear canvases
                document.getElementById('patchOverlay').style.display = 'none';
                
                const outputCanvas = document.getElementById('patchCanvas');
                const outputCtx = outputCanvas.getContext('2d');
                outputCtx.clearRect(0, 0, outputCanvas.width, outputCanvas.height);
                
                const liveCanvas = document.getElementById('liveCanvas');
                const liveCtx = liveCanvas.getContext('2d');
                liveCtx.clearRect(0, 0, liveCanvas.width, liveCanvas.height);
                
                // Clear video
                document.getElementById('patchVideo').srcObject = null;
            }
            
            function showPatchOverlay(element) {
                const overlay = document.getElementById('patchOverlay');
                
                // Calculate center patch position based on element size
                const patchPixelSize = (patchSize / 640) * element.offsetWidth; // Scale to display size
                const centerX = (element.offsetWidth - patchPixelSize) / 2;
                const centerY = (element.offsetHeight - patchPixelSize) / 2;
                
                overlay.style.display = 'block';
                overlay.style.left = centerX + 'px';
                overlay.style.top = centerY + 'px';
                overlay.style.width = patchPixelSize + 'px';
                overlay.style.height = patchPixelSize + 'px';
            }
            
            function toggleStreamThrottle() {
                const video = document.getElementById('patchVideo');
                const canvas = document.getElementById('liveCanvas');
                const button = document.getElementById('throttleToggle');
                const choppinessSlider = document.getElementById('choppinessSlider');
                const choppinessValue = document.getElementById('choppinessValue');
                const choppinessLabel = document.getElementById('choppinessLabel');
                
                streamThrottleEnabled = !streamThrottleEnabled;
                
                if (streamThrottleEnabled) {
                    // Enable choppy bandwidth simulation - switch to canvas
                    video.style.display = 'none';
                    canvas.style.display = 'block';
                    choppinessLabel.style.display = 'inline';
                    choppinessSlider.style.display = 'inline';
                    choppinessValue.style.display = 'inline';
                    button.textContent = '📺 Disable Choppy Stream';
                    button.className = 'danger';
                    
                    // Move overlay to canvas and start choppy renderer
                    showPatchOverlay(canvas);
                    startStreamRenderer();
                } else {
                    // Disable choppy effect - switch back to smooth video
                    canvas.style.display = 'none';
                    video.style.display = 'block';
                    choppinessLabel.style.display = 'none';
                    choppinessSlider.style.display = 'none';
                    choppinessValue.style.display = 'none';
                    button.textContent = '📺 Enable Choppy Stream';
                    button.className = '';
                    
                    // Stop choppy renderer and move overlay back to video
                    stopStreamRenderer();
                    showPatchOverlay(video);
                }
            }
            
            function updateChoppiness(value) {
                choppinessLevel = parseInt(value);
                const qualityLabels = {
                    1: "Terrible", 2: "Very Low", 3: "Low", 4: "Poor", 5: "Fair",
                    6: "OK", 7: "Good", 8: "High", 9: "Very High", 10: "Perfect"
                };
                document.getElementById('choppinessValue').textContent = qualityLabels[choppinessLevel];
                
                // Restart renderer if it's enabled
                if (streamThrottleEnabled) {
                    startStreamRenderer();
                }
            }
            
            function startStreamRenderer() {
                stopStreamRenderer(); // Clear any existing renderer
                
                const video = document.getElementById('patchVideo');
                const canvas = document.getElementById('liveCanvas');
                const ctx = canvas.getContext('2d');
                
                // Use same frame rate as VAE processing
                const frameInterval = 1000 / patchProcessingFPS;
                
                streamRenderInterval = setInterval(async () => {
                    if (video && video.videoWidth > 0 && video.readyState >= 2) {
                        try {
                            // Capture frame from video
                            const frameCanvas = document.createElement('canvas');
                            frameCanvas.width = 640;
                            frameCanvas.height = 480;
                            const frameCtx = frameCanvas.getContext('2d');
                            frameCtx.drawImage(video, 0, 0, 640, 480);
                            
                            // Convert frame to blob for API upload
                            frameCanvas.toBlob(async (blob) => {
                                if (blob) {
                                    const formData = new FormData();
                                    formData.append('file', blob, 'frame.jpg');
                                    formData.append('choppiness_level', choppinessLevel.toString());
                                    
                                    try {
                                        const response = await fetch('/apply_choppiness', {
                                            method: 'POST',
                                            body: formData
                                        });
                                        
                                        if (response.ok) {
                                            const result = await response.json();
                                            if (result.success) {
                                                // Display choppy frame on canvas
                                                const choppyImg = new Image();
                                                choppyImg.onload = () => {
                                                    ctx.clearRect(0, 0, canvas.width, canvas.height);
                                                    ctx.drawImage(choppyImg, 0, 0, canvas.width, canvas.height);
                                                };
                                                choppyImg.src = 'data:image/jpeg;base64,' + result.choppy_image_b64;
                                            }
                                        }
                                    } catch (apiError) {
                                        console.error('Choppiness API error:', apiError);
                                        // Fallback: show original frame
                                        ctx.clearRect(0, 0, canvas.width, canvas.height);
                                        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                                    }
                                }
                            }, 'image/jpeg', 0.8);
                            
                            patchInputFPSCounter++;
                        } catch (error) {
                            console.error('Stream render error:', error);
                        }
                    }
                }, frameInterval);
            }
            
            function stopStreamRenderer() {
                if (streamRenderInterval) {
                    clearInterval(streamRenderInterval);
                    streamRenderInterval = null;
                }
            }
            
            function togglePatchProcessing() {
                isPatchProcessing = !isPatchProcessing;
                const btn = document.getElementById('patchProcessToggle');
                const indicator = document.getElementById('processingIndicator');
                
                if (isPatchProcessing) {
                    btn.textContent = '⏸️ Stop Processing';
                    btn.className = 'danger';
                    indicator.style.display = 'block';
                    startPatchProcessingLoop();
                } else {
                    btn.textContent = '🚀 Start VAE Processing';
                    btn.className = '';
                    indicator.style.display = 'none';
                }
            }
            
            function updatePatchFPS(value) {
                patchProcessingFPS = parseInt(value);
                document.getElementById('patchFpsValue').textContent = value;
            }
            
            function updatePatchSize() {
                patchSize = parseInt(document.getElementById('patchSizeSelect').value);
                // Update overlay on whichever element is visible
                const video = document.getElementById('patchVideo');
                const canvas = document.getElementById('liveCanvas');
                
                if (streamThrottleEnabled && canvas.style.display !== 'none') {
                    showPatchOverlay(canvas);
                } else if (video.srcObject) {
                    showPatchOverlay(video);
                }
            }
            
            async function startPatchProcessingLoop() {
                const video = document.getElementById('patchVideo'); // Hidden video for capture
                const outputCanvas = document.getElementById('patchCanvas'); // For processed output
                const ctx = outputCanvas.getContext('2d');
                
                const frameInterval = 1000 / patchProcessingFPS;
                
                async function processPatch() {
                    if (!isPatchProcessing) return;
                    
                    try {
                        // Capture full frame from video
                        const tempCanvas = document.createElement('canvas');
                        tempCanvas.width = 640;
                        tempCanvas.height = 480;
                        const tempCtx = tempCanvas.getContext('2d');
                        tempCtx.drawImage(video, 0, 0, 640, 480);
                        
                        // Convert to blob
                        const blob = await new Promise(resolve => {
                            tempCanvas.toBlob(resolve, 'image/jpeg', 0.8);
                        });
                        
                        // Send to patch processing endpoint
                        const formData = new FormData();
                        formData.append('file', blob, 'frame.jpg');
                        
                        const startTime = performance.now();
                        const response = await fetch('/process_patch_custom', {
                            method: 'POST',
                            body: formData
                        });
                        
                        if (response.ok) {
                            const result = await response.json();
                            const processTime = performance.now() - startTime;
                            
                            // Update performance stats
                            document.getElementById('patchEncodeTime').textContent = result.timing.encode_ms;
                            document.getElementById('patchDecodeTime').textContent = result.timing.decode_ms;
                            document.getElementById('patchTotalTime').textContent = Math.round(processTime);
                            document.getElementById('patchFramesProcessed').textContent = ++patchFramesProcessed;
                            
                            // Show processed patch in sidebar
                            document.getElementById('patchPreview').src = 'data:image/jpeg;base64,' + result.processed_patch_b64;
                            
                            // Composite result back to canvas (processed output)
                            const img = new Image();
                            img.onload = () => {
                                // Clear canvas and draw composited result
                                ctx.clearRect(0, 0, outputCanvas.width, outputCanvas.height);
                                ctx.drawImage(img, 0, 0, outputCanvas.width, outputCanvas.height);
                                patchOutputFPSCounter++;
                            };
                            img.src = 'data:image/jpeg;base64,' + result.composited_image_b64;
                        }
                        
                    } catch (error) {
                        console.error('Patch processing error:', error);
                    }
                    
                    // Schedule next frame
                    if (isPatchProcessing) {
                        setTimeout(processPatch, frameInterval);
                    }
                }
                
                processPatch();
            }
            
            function startPatchFPSCounters() {
                // Simple FPS counter for video mode
                function countVideoFrames() {
                    const video = document.getElementById('patchVideo');
                    if (video && video.videoWidth > 0 && !streamThrottleEnabled) {
                        patchInputFPSCounter++;
                    }
                    requestAnimationFrame(countVideoFrames);
                }
                countVideoFrames();
                
                // Update FPS displays every second
                setInterval(() => {
                    document.getElementById('patchInputFPS').textContent = patchInputFPSCounter + ' FPS';
                    document.getElementById('patchOutputFPS').textContent = patchOutputFPSCounter + ' FPS';
                    patchInputFPSCounter = 0;
                    patchOutputFPSCounter = 0;
                }, 1000);
            }
            
            // Image upload functions (existing)
            let selectedFile = null;
            
            document.getElementById('fileInput').addEventListener('change', function(e) {
                selectedFile = e.target.files[0];
                if (selectedFile) {
                    document.getElementById('processBtn').disabled = false;
                    const reader = new FileReader();
                    reader.onload = function(e) {
                        document.getElementById('originalImg').src = e.target.result;
                    };
                    reader.readAsDataURL(selectedFile);
                }
            });
            
            async function processImage() {
                if (!selectedFile) return;
                
                const formData = new FormData();
                formData.append('file', selectedFile);
                
                // Get selected processor
                const selectedProcessor = document.querySelector('input[name="imageProcessor"]:checked').value;
                formData.append('processor', selectedProcessor);
                
                document.getElementById('processBtn').disabled = true;
                document.getElementById('processBtn').textContent = `⚡ Processing (${selectedProcessor})...`;
                
                try {
                    const response = await fetch('/process_patch_combined', {
                        method: 'POST',
                        body: formData
                    });
                    
                    if (!response.ok) throw new Error('Processing failed');
                    
                    const result = await response.json();
                    
                    // Display results - handle different response formats
                    const imageB64 = result.output_image_b64 || result.composited_image_b64;
                    document.getElementById('decodedImg').src = 'data:image/jpeg;base64,' + imageB64;
                    document.getElementById('timing').textContent = 
                        `⚡ Processor: ${selectedProcessor} | Encode: ${result.timing.encode_ms}ms | Decode: ${result.timing.decode_ms}ms | Total: ${result.timing.total_ms}ms`;
                    document.getElementById('stats').textContent = JSON.stringify(result.latent_stats, null, 2);
                    document.getElementById('results').style.display = 'block';
                    
                } catch (error) {
                    alert('Error: ' + error.message);
                } finally {
                    document.getElementById('processBtn').disabled = false;
                    document.getElementById('processBtn').textContent = '🚀 Process Image';
                }
            }
        </script>
    </body>
    </html>
    """

@app.post("/process")
async def process_image(file: UploadFile = File(...)):
    """Process uploaded image through VAE encode-decode pipeline"""
    if not processor:
        raise HTTPException(status_code=500, detail="VAE processor not initialized")
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read uploaded image
        contents = await file.read()
        pil_image = Image.open(io.BytesIO(contents))
        
        # Process image
        result = processor.encode_decode(pil_image)
        
        # Convert output image to base64
        img_buffer = io.BytesIO()
        result["output_image"].save(img_buffer, format="JPEG", quality=95)
        img_b64 = base64.b64encode(img_buffer.getvalue()).decode()
        
        return JSONResponse({
            "success": True,
            "output_image_b64": img_b64,
            "latent_stats": result["latent_stats"],
            "timing": result["timing"]
        })
        
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.post("/process_patch")
async def process_patch(file: UploadFile = File(...)):
    """Process a patch from full frame and return composited result"""
    if not processor:
        raise HTTPException(status_code=500, detail="VAE processor not initialized")
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read uploaded full frame
        contents = await file.read()
        full_frame = Image.open(io.BytesIO(contents))
        
        # Extract center patch (256x256 from center of frame)
        frame_width, frame_height = full_frame.size
        
        # Calculate center patch coordinates
        patch_size = 256
        center_x = frame_width // 2
        center_y = frame_height // 2
        
        # Crop center patch
        left = max(0, center_x - patch_size // 2)
        top = max(0, center_y - patch_size // 2)
        right = min(frame_width, center_x + patch_size // 2)
        bottom = min(frame_height, center_y + patch_size // 2)
        
        patch = full_frame.crop((left, top, right, bottom))
        
        # Resize patch to exactly 256x256 if needed
        if patch.size != (256, 256):
            patch = patch.resize((256, 256), Image.BICUBIC)
        
        # Process patch through VAE
        result = processor.encode_decode(patch)
        processed_patch = result["output_image"]
        
        # Create composited image
        composited = full_frame.copy()
        
        # Resize processed patch back to original patch size if needed
        original_patch_size = (right - left, bottom - top)
        if processed_patch.size != original_patch_size:
            processed_patch = processed_patch.resize(original_patch_size, Image.BICUBIC)
        
        # Paste processed patch back
        composited.paste(processed_patch, (left, top))
        
        # Convert to base64
        img_buffer = io.BytesIO()
        composited.save(img_buffer, format="JPEG", quality=95)
        composited_b64 = base64.b64encode(img_buffer.getvalue()).decode()
        
        # Also return the patch info for debugging/visualization
        patch_buffer = io.BytesIO()
        processed_patch.save(patch_buffer, format="JPEG", quality=95)
        patch_b64 = base64.b64encode(patch_buffer.getvalue()).decode()
        
        return JSONResponse({
            "success": True,
            "composited_image_b64": composited_b64,
            "processed_patch_b64": patch_b64,
            "patch_coords": {
                "left": left,
                "top": top,
                "right": right,
                "bottom": bottom,
                "center_x": center_x,
                "center_y": center_y
            },
            "latent_stats": result["latent_stats"],
            "timing": result["timing"]
        })
        
    except Exception as e:
        logger.error(f"Error processing patch: {e}")
        raise HTTPException(status_code=500, detail=f"Patch processing failed: {str(e)}")

@app.post("/apply_choppiness")
async def apply_choppiness(file: UploadFile = File(...), choppiness_level: int = 3):
    """Apply bandwidth simulation choppiness to image"""
    try:
        # Read uploaded image
        img_data = await file.read()
        input_image = Image.open(io.BytesIO(img_data)).convert('RGB')
        
        # Calculate compression factor (1-10 scale)
        compression_factor = max(1, 11 - choppiness_level)
        
        # Get original dimensions
        orig_width, orig_height = input_image.size
        
        # Calculate low-res dimensions 
        low_width = max(32, orig_width // compression_factor)
        low_height = max(24, orig_height // compression_factor)
        
        # Downscale (creates pixelation)
        low_res = input_image.resize((low_width, low_height), Image.NEAREST)
        
        # Upscale back (maintains pixelated look)
        choppy_image = low_res.resize((orig_width, orig_height), Image.NEAREST)
        
        # Convert to base64
        buffer = io.BytesIO()
        choppy_image.save(buffer, format='JPEG', quality=90)
        buffer.seek(0)
        choppy_b64 = base64.b64encode(buffer.getvalue()).decode()
        
        return JSONResponse({
            "success": True,
            "choppy_image_b64": choppy_b64,
            "compression_factor": compression_factor,
            "original_size": f"{orig_width}x{orig_height}",
            "compressed_size": f"{low_width}x{low_height}"
        })
        
    except Exception as e:
        logger.error(f"Error applying choppiness: {e}")
        raise HTTPException(status_code=500, detail=f"Choppiness processing failed: {str(e)}")

@app.post("/encode_latent")
async def encode_latent(file: UploadFile = File(...)):
    """Encode image to latent vector for transmission"""
    if not processor:
        raise HTTPException(status_code=500, detail="VAE processor not initialized")
    
    try:
        # Read uploaded image
        img_data = await file.read()
        input_image = Image.open(io.BytesIO(img_data)).convert('RGB')
        
        # Preprocess image using processor's method
        import numpy as np
        input_array = processor.preprocess_image(input_image)
        
        # Encode to latent space
        start_time = time.time()
        enc_out = processor.enc_model.predict({"x": input_array})
        encode_time = (time.time() - start_time) * 1000
        
        # Extract latent
        latent = next(iter(enc_out.values()))
        latent = np.array(latent)
        
        # Ensure NCHW format
        if latent.shape[1] != 4:
            latent = np.transpose(latent, (0, 3, 1, 2))
        
        # Debug: Log original array properties before serialization
        logger.info(f"Original latent before serialization: shape={latent.shape}, dtype={latent.dtype}, "
                   f"contiguous={latent.flags.c_contiguous}, strides={latent.strides}, "
                   f"itemsize={latent.itemsize}, nbytes={latent.nbytes}")
        
        # Convert latent to base64 for transmission using numpy's serialization
        # This preserves array metadata better than tobytes()
        try:
            logger.info("Creating BytesIO buffer for latent serialization...")
            latent_buffer = io.BytesIO()
            logger.info("Saving latent to buffer with np.save...")
            np.save(latent_buffer, latent)
            logger.info("Getting bytes from buffer...")
            latent_bytes = latent_buffer.getvalue()
            logger.info(f"Encoding to base64, bytes length: {len(latent_bytes)}")
            latent_b64 = base64.b64encode(latent_bytes).decode()
            logger.info(f"Base64 encoding complete, length: {len(latent_b64)}")
        except Exception as e:
            logger.error(f"Error in latent serialization: {e}")
            logger.error(f"io module: {io}")
            logger.error(f"io.BytesIO: {io.BytesIO}")
            raise
        
        return JSONResponse({
            "success": True,
            "latent_b64": latent_b64,
            "latent_shape": list(latent.shape),
            "latent_dtype": str(latent.dtype),
            "encode_time_ms": round(encode_time, 2),
            "latent_stats": {
                "mean": float(latent.mean()),
                "std": float(latent.std()),
                "min": float(latent.min()),
                "max": float(latent.max())
            }
        })
        
    except Exception as e:
        logger.error(f"Error encoding latent: {e}")
        raise HTTPException(status_code=500, detail=f"Latent encoding failed: {str(e)}")

@app.post("/encode_latent_16p")
async def encode_latent_16p(file: UploadFile = File(...)):
    """Encode image to compressed 16x16 latent vector (16P resolution)"""
    if not processor:
        raise HTTPException(status_code=500, detail="VAE processor not initialized")
    
    try:
        # Read uploaded image
        img_data = await file.read()
        input_image = Image.open(io.BytesIO(img_data)).convert('RGB')
        
        # Preprocess image using processor's method
        import numpy as np
        input_array = processor.preprocess_image(input_image)
        
        # Encode to latent space
        start_time = time.time()
        enc_out = processor.enc_model.predict({"x": input_array})
        encode_time = (time.time() - start_time) * 1000
        
        # Extract latent
        latent = next(iter(enc_out.values()))
        latent = np.array(latent)
        
        # Ensure NCHW format
        if latent.shape[1] != 4:
            latent = np.transpose(latent, (0, 3, 1, 2))
        
        logger.info(f"Original latent shape before 16P downscaling: {latent.shape}")
        
        # Downscale latent to 16x16 resolution (16P)
        # Original latent is typically [1, 4, 32, 32] for 256x256 input
        # We want to compress it to [1, 4, 16, 16] for reduced bandwidth with better quality than 8P
        from scipy.ndimage import zoom
        
        # Calculate zoom factors to get to 16x16
        current_h, current_w = latent.shape[2], latent.shape[3]
        zoom_h = 16.0 / current_h
        zoom_w = 16.0 / current_w
        
        # Downscale using bilinear interpolation for better quality than 8P
        latent_16p = zoom(latent, (1, 1, zoom_h, zoom_w), order=1)  # order=1 = bilinear
        
        # Ensure exact 16x16 size
        if latent_16p.shape[2:] != (16, 16):
            # Force reshape to exactly 16x16 if zoom didn't give exact result
            latent_16p = latent_16p[:, :, :16, :16]
        
        logger.info(f"16P latent shape after downscaling: {latent_16p.shape}")
        logger.info(f"16P latent size reduction: {latent.nbytes} -> {latent_16p.nbytes} bytes ({100.0 * latent_16p.nbytes / latent.nbytes:.1f}%)")
        
        # Serialize 16P latent
        latent_buffer = io.BytesIO()
        np.save(latent_buffer, latent_16p)
        latent_bytes = latent_buffer.getvalue()
        latent_b64 = base64.b64encode(latent_bytes).decode()
        
        return JSONResponse({
            "success": True,
            "latent_b64": latent_b64,
            "latent_shape": list(latent_16p.shape),
            "latent_dtype": str(latent_16p.dtype),
            "encode_time_ms": round(encode_time, 2),
            "original_size_bytes": int(latent.nbytes),
            "compressed_size_bytes": int(latent_16p.nbytes),
            "compression_ratio": round(latent.nbytes / latent_16p.nbytes, 1),
            "latent_stats": {
                "mean": float(latent_16p.mean()),
                "std": float(latent_16p.std()),
                "min": float(latent_16p.min()),
                "max": float(latent_16p.max())
            }
        })
        
    except Exception as e:
        logger.error(f"Error encoding 16P latent: {e}")
        raise HTTPException(status_code=500, detail=f"16P latent encoding failed: {str(e)}")

@app.post("/decode_latent")
async def decode_latent(request: dict):
    """Decode latent vector back to image"""
    if not processor:
        raise HTTPException(status_code=500, detail="VAE processor not initialized")
    
    try:
        # Extract parameters from request
        latent_b64 = request.get("latent_b64")
        
        if not latent_b64:
            raise HTTPException(status_code=400, detail="Missing latent_b64")
        
        # Convert base64 back to numpy array using numpy's deserialization
        # This preserves the original array metadata
        latent_bytes = base64.b64decode(latent_b64)
        
        # Reconstruct latent array with preserved metadata
        latent_buffer = io.BytesIO(latent_bytes)
        latent = np.load(latent_buffer)
        
        logger.info(f"Reconstructed latent with numpy.load: shape={latent.shape}, dtype={latent.dtype}, "
                   f"contiguous={latent.flags.c_contiguous}, strides={latent.strides}, "
                   f"itemsize={latent.itemsize}, nbytes={latent.nbytes}")
        
        # The reconstructed array should now be identical to the original
        # No additional processing needed since numpy.save/load preserves everything
        
        # Decode from latent space
        start_time = time.time()
        dec_out = processor.dec_model.predict({"z_scaled": latent})
        decode_time = (time.time() - start_time) * 1000
        
        # Extract and postprocess output
        output = next(iter(dec_out.values()))
        output = np.array(output)
        output_01 = processor.postprocess_output(output)
        
        # Convert back to PIL Image
        output_hwc = (output_01[0].transpose(1, 2, 0) * 255).astype(np.uint8)
        output_image = Image.fromarray(output_hwc)
        
        # Convert to base64
        buffer = io.BytesIO()
        output_image.save(buffer, format='JPEG', quality=90)
        buffer.seek(0)
        image_b64 = base64.b64encode(buffer.getvalue()).decode()
        
        return JSONResponse({
            "success": True,
            "image_b64": image_b64,
            "decode_time_ms": round(decode_time, 2),
            "output_size": f"{output_image.width}x{output_image.height}"
        })
        
    except Exception as e:
        logger.error(f"Error decoding latent: {e}")
        raise HTTPException(status_code=500, detail=f"Latent decoding failed: {str(e)}")

@app.post("/decode_latent_16p")
async def decode_latent_16p(request: dict):
    """Decode compressed 16x16 latent vector (16P resolution) back to image"""
    if not processor:
        raise HTTPException(status_code=500, detail="VAE processor not initialized")
    
    try:
        # Extract parameters from request
        latent_b64 = request.get("latent_b64")
        
        if not latent_b64:
            raise HTTPException(status_code=400, detail="Missing latent_b64")
        
        # Convert base64 back to numpy array
        latent_bytes = base64.b64decode(latent_b64)
        latent_buffer = io.BytesIO(latent_bytes)
        latent_16p = np.load(latent_buffer)
        
        logger.info(f"Reconstructed 16P latent: shape={latent_16p.shape}, dtype={latent_16p.dtype}")
        
        # Upscale 16P latent back to full resolution for decoder
        # The VAE decoder expects [1, 4, 32, 32] for 256x256 output
        from scipy.ndimage import zoom
        
        target_h, target_w = 32, 32  # Standard latent size for 256x256 input
        current_h, current_w = latent_16p.shape[2], latent_16p.shape[3]
        
        zoom_h = target_h / current_h
        zoom_w = target_w / current_w
        
        # Upscale using bilinear interpolation for better quality
        latent_upscaled = zoom(latent_16p, (1, 1, zoom_h, zoom_w), order=1)  # order=1 = bilinear
        
        # Ensure exact target size
        if latent_upscaled.shape[2:] != (target_h, target_w):
            # Crop or pad to exact size if needed
            latent_full = np.zeros((1, 4, target_h, target_w), dtype=latent_16p.dtype)
            h_crop = min(target_h, latent_upscaled.shape[2])
            w_crop = min(target_w, latent_upscaled.shape[3])
            latent_full[:, :, :h_crop, :w_crop] = latent_upscaled[:, :, :h_crop, :w_crop]
        else:
            latent_full = latent_upscaled
            
        logger.info(f"16P latent upscaled from {latent_16p.shape} to {latent_full.shape}")
        
        # Decode from latent space
        start_time = time.time()
        dec_out = processor.dec_model.predict({"z_scaled": latent_full})
        decode_time = (time.time() - start_time) * 1000
        
        # Extract and postprocess output
        output = next(iter(dec_out.values()))
        output = np.array(output)
        output_01 = processor.postprocess_output(output)
        
        # Convert back to PIL Image
        output_hwc = (output_01[0].transpose(1, 2, 0) * 255).astype(np.uint8)
        output_image = Image.fromarray(output_hwc)
        
        # Convert to base64
        buffer = io.BytesIO()
        output_image.save(buffer, format='JPEG', quality=90)
        buffer.seek(0)
        image_b64 = base64.b64encode(buffer.getvalue()).decode()
        
        return JSONResponse({
            "success": True,
            "image_b64": image_b64,
            "decode_time_ms": round(decode_time, 2),
            "output_size": f"{output_image.width}x{output_image.height}",
            "upscaled_from": f"{latent_16p.shape[2]}x{latent_16p.shape[3]}",
            "upscaled_to": f"{latent_full.shape[2]}x{latent_full.shape[3]}"
        })
        
    except Exception as e:
        logger.error(f"Error decoding 16P latent: {e}")
        raise HTTPException(status_code=500, detail=f"16P latent decoding failed: {str(e)}")

@app.post("/process_custom")
async def process_custom(file: UploadFile = File(...)):
    """Process uploaded image through custom PyTorch AE encode-decode pipeline"""
    if not custom_processor:
        if not TORCH_AVAILABLE:
            raise HTTPException(status_code=503, detail="PyTorch not available - custom AE disabled")
        else:
            raise HTTPException(status_code=503, detail="Custom AE processor not initialized")
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read uploaded image
        contents = await file.read()
        pil_image = Image.open(io.BytesIO(contents))
        
        # Process image through custom AE
        result = custom_processor.encode_decode(pil_image)
        
        # Convert output image to base64
        img_buffer = io.BytesIO()
        result["output_image"].save(img_buffer, format="JPEG", quality=95)
        img_b64 = base64.b64encode(img_buffer.getvalue()).decode()
        
        return JSONResponse({
            "success": True,
            "output_image_b64": img_b64,
            "latent_stats": result["latent_stats"],
            "timing": result["timing"],
            "processor_type": "custom_pytorch"
        })
        
    except Exception as e:
        logger.error(f"Error processing image with custom AE: {e}")
        raise HTTPException(status_code=500, detail=f"Custom processing failed: {str(e)}")

@app.post("/process_patch_custom")
async def process_patch_custom(file: UploadFile = File(...)):
    """Process a patch from full frame using custom AE and return composited result"""
    if not custom_processor:
        if not TORCH_AVAILABLE:
            raise HTTPException(status_code=503, detail="PyTorch not available - custom AE disabled")
        else:
            raise HTTPException(status_code=503, detail="Custom AE processor not initialized")
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read uploaded full frame
        contents = await file.read()
        full_frame = Image.open(io.BytesIO(contents))
        
        # Extract center patch (256x256 from center of frame)
        frame_width, frame_height = full_frame.size
        
        # Calculate center patch coordinates
        patch_size = 256
        center_x = frame_width // 2
        center_y = frame_height // 2
        
        # Crop center patch
        left = max(0, center_x - patch_size // 2)
        top = max(0, center_y - patch_size // 2)
        right = min(frame_width, center_x + patch_size // 2)
        bottom = min(frame_height, center_y + patch_size // 2)
        
        patch = full_frame.crop((left, top, right, bottom))
        
        # Resize patch to exactly 256x256 if needed
        if patch.size != (256, 256):
            patch = patch.resize((256, 256), Image.BICUBIC)
        
        # Process patch through custom AE
        result = custom_processor.encode_decode(patch)
        processed_patch = result["output_image"]
        
        # Create composited image
        composited = full_frame.copy()
        
        # Resize processed patch back to original patch size if needed
        original_patch_size = (right - left, bottom - top)
        if processed_patch.size != original_patch_size:
            processed_patch = processed_patch.resize(original_patch_size, Image.BICUBIC)
        
        # Paste processed patch back
        composited.paste(processed_patch, (left, top))
        
        # Convert to base64
        img_buffer = io.BytesIO()
        composited.save(img_buffer, format="JPEG", quality=95)
        composited_b64 = base64.b64encode(img_buffer.getvalue()).decode()
        
        # Also return the patch info for debugging/visualization
        patch_buffer = io.BytesIO()
        processed_patch.save(patch_buffer, format="JPEG", quality=95)
        patch_b64 = base64.b64encode(patch_buffer.getvalue()).decode()
        
        return JSONResponse({
            "success": True,
            "composited_image_b64": composited_b64,
            "processed_patch_b64": patch_b64,
            "patch_coords": {
                "left": left,
                "top": top,
                "right": right,
                "bottom": bottom,
                "center_x": center_x,
                "center_y": center_y
            },
            "latent_stats": result["latent_stats"],
            "timing": result["timing"],
            "processor_type": "custom_pytorch"
        })
        
    except Exception as e:
        logger.error(f"Error processing patch with custom AE: {e}")
        raise HTTPException(status_code=500, detail=f"Custom patch processing failed: {str(e)}")

@app.post("/process_patch_combined")
async def process_patch_combined(file: UploadFile = File(...), processor: str = Form(default="custom")):
    """Combined endpoint for testing both processors - custom (default) or coreml"""
    if processor == "custom":
        # Use custom AE processor
        if not custom_processor:
            if not TORCH_AVAILABLE:
                raise HTTPException(status_code=503, detail="PyTorch not available - custom AE disabled")
            else:
                raise HTTPException(status_code=503, detail="Custom AE processor not initialized")
        return await process_patch_custom(file)
    
    elif processor == "coreml":
        # Use CoreML VAE processor (global variable)
        if not globals()['processor']:
            raise HTTPException(status_code=503, detail="CoreML VAE processor not initialized")
        return await process_patch(file)
    
    else:
        raise HTTPException(status_code=400, detail="Invalid processor type. Use 'custom' or 'coreml'")

@app.post("/encode_latent_custom")
async def encode_latent_custom(file: UploadFile = File(...)):
    """Encode image to latent using custom AE - compatible with WebRTC transmission"""
    if not custom_processor:
        if not TORCH_AVAILABLE:
            raise HTTPException(status_code=503, detail="PyTorch not available - custom AE disabled")
        else:
            raise HTTPException(status_code=503, detail="Custom AE processor not initialized")
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read and process image
        contents = await file.read()
        pil_image = Image.open(io.BytesIO(contents))
        
        # Get latent representation from custom AE
        input_tensor = custom_processor.to_tensor(pil_image)
        with torch.no_grad():
            latent = custom_processor.encoder(input_tensor)
        
        # Convert latent to base64 for transmission
        latent_np = latent.cpu().numpy()
        latent_bytes = latent_np.tobytes()
        latent_b64 = base64.b64encode(latent_bytes).decode()
        
        return JSONResponse({
            "success": True,
            "latent_b64": latent_b64,
            "latent_shape": list(latent_np.shape),
            "latent_dtype": str(latent_np.dtype),
            "processor_type": "custom_pytorch"
        })
        
    except Exception as e:
        logger.error(f"Error encoding with custom AE: {e}")
        raise HTTPException(status_code=500, detail=f"Custom encoding failed: {str(e)}")

@app.post("/decode_latent_custom")
async def decode_latent_custom(request: dict):
    """Decode latent using custom AE - compatible with WebRTC transmission"""
    if not custom_processor:
        if not TORCH_AVAILABLE:
            raise HTTPException(status_code=503, detail="PyTorch not available - custom AE disabled")
        else:
            raise HTTPException(status_code=503, detail="Custom AE processor not initialized")
    
    try:
        # Decode latent from base64
        latent_b64 = request.get("latent_b64")
        latent_shape = request.get("latent_shape")
        latent_dtype = request.get("latent_dtype", "float32")
        
        if not latent_b64 or not latent_shape:
            raise ValueError("Missing latent_b64 or latent_shape")
        
        # Reconstruct latent tensor
        latent_bytes = base64.b64decode(latent_b64)
        latent_np = np.frombuffer(latent_bytes, dtype=latent_dtype).reshape(latent_shape)
        latent_tensor = torch.from_numpy(latent_np).to(custom_processor.device)
        
        # Decode using custom AE
        with torch.no_grad():
            decoded_tensor = custom_processor.decoder(latent_tensor)
        
        # Convert to PIL and base64
        output_pil = custom_processor.to_pil(decoded_tensor.squeeze().cpu())
        img_buffer = io.BytesIO()
        output_pil.save(img_buffer, format="JPEG", quality=95)
        img_b64 = base64.b64encode(img_buffer.getvalue()).decode()
        
        return JSONResponse({
            "success": True,
            "output_image_b64": img_b64,
            "processor_type": "custom_pytorch"
        })
        
    except Exception as e:
        logger.error(f"Error decoding with custom AE: {e}")
        raise HTTPException(status_code=500, detail=f"Custom decoding failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "coreml_processor_ready": processor is not None,
        "custom_processor_ready": custom_processor is not None,
        "pytorch_available": TORCH_AVAILABLE,
        "timestamp": time.time()
    }

if __name__ == "__main__":
    uvicorn.run(
        "vae_server:app", 
        host="127.0.0.1", 
        port=8000, 
        reload=False,  # Don't reload to keep models in memory
        log_level="info"
    )