#!/bin/bash

echo "🚀 Starting VAE Processing Server..."
echo "📦 Installing dependencies..."

# Install requirements if needed
pip install -r requirements_server.txt

echo "🔥 Starting server with warm models..."
echo "📡 Server will be available at: http://127.0.0.1:8000"
echo "💡 Use the web interface or API endpoints"

python vae_server.py