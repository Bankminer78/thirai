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
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

# Initialize processor globally (loaded once at startup)
logger.info("🚀 Initializing VAE Processor...")
try:
    processor = VAEProcessor(size=256)  # Change size as needed
    logger.info("✅ VAE Processor initialized successfully!")
except Exception as e:
    logger.error(f"❌ Failed to initialize VAE Processor: {e}")
    processor = None

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
        </style>
    </head>
    <body>
        <h1>🎥🔥 VAE Live Video Processor</h1>
        <p>Real-time video processing with instant VAE encoding/decoding</p>
        
        <div class="tabs">
            <div class="tab active" onclick="switchTab('video')">🎥 Live Video</div>
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
        
        <!-- Image Upload Tab -->
        <div id="image-tab" class="tab-content">
            <div class="upload-area" onclick="document.getElementById('fileInput').click()">
                <input type="file" id="fileInput" accept="image/*" style="display: none">
                📁 Click to select image or drag & drop
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
                        
                        const startTime = performance.now();
                        const response = await fetch('/process', {
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
                            
                            // Display processed frame
                            const img = new Image();
                            img.onload = () => {
                                ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
                                outputFPSCounter++;
                            };
                            img.src = 'data:image/jpeg;base64,' + result.output_image_b64;
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
                
                document.getElementById('processBtn').disabled = true;
                document.getElementById('processBtn').textContent = '⚡ Processing...';
                
                try {
                    const response = await fetch('/process', {
                        method: 'POST',
                        body: formData
                    });
                    
                    if (!response.ok) throw new Error('Processing failed');
                    
                    const result = await response.json();
                    
                    // Display results
                    document.getElementById('decodedImg').src = 'data:image/jpeg;base64,' + result.output_image_b64;
                    document.getElementById('timing').textContent = 
                        `⚡ Encode: ${result.timing.encode_ms}ms | Decode: ${result.timing.decode_ms}ms | Total: ${result.timing.total_ms}ms`;
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

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "processor_ready": processor is not None,
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