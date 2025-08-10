#!/bin/bash

echo "🚀 Starting WebRTC Super Resolution App..."
echo ""
echo "📋 Prerequisites:"
echo "  - Python 3.8+ installed"
echo "  - macOS (for CoreML models)"
echo "  - Camera access enabled"
echo ""

# Check if requirements are installed
if ! python3 -c "import coremltools, fastapi, PIL" 2>/dev/null; then
    echo "📦 Installing Python dependencies..."
    pip install -r requirements_server.txt
    echo ""
fi

echo "🔥 Starting VAE server on http://127.0.0.1:8000..."
echo ""
echo "🌐 After server starts:"
echo "  1. Open conf.html in your web browser"
echo "  2. Choose SR Sender or SR Receiver role"
echo "  3. Enable camera when prompted"
echo "  4. Connect with another peer for video chat + SR"
echo ""
echo "⏹️  Press Ctrl+C to stop the server"
echo "================================================="

python3 vae_server.py