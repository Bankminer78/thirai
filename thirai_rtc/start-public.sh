#!/bin/bash

# Check if ngrok is installed
if ! command -v ngrok &> /dev/null; then
    echo "🚨 ngrok not found! Installing..."
    if command -v brew &> /dev/null; then
        brew install ngrok
    else
        echo "Please install ngrok from https://ngrok.com/download"
        exit 1
    fi
fi

# Check if ngrok is authenticated
if ! ngrok config check &> /dev/null; then
    echo "🔐 First time setup: Please authenticate ngrok"
    echo "1. Go to: https://dashboard.ngrok.com/get-started/your-authtoken"
    echo "2. Copy your authtoken"
    echo "3. Run: ngrok config add-authtoken YOUR_TOKEN"
    echo ""
    read -p "Press Enter after setting up your authtoken..."
fi

echo "🚀 Starting Thirai RTC with public access..."
echo "📡 Server will start on port 3000"
echo "🌐 Ngrok will create a public tunnel"
echo ""

# Start server in background
node server/index.js &
SERVER_PID=$!

# Wait a moment for server to start
sleep 2

# Start ngrok and capture output
echo "🌐 Starting ngrok tunnel..."
ngrok http 3000 --log stdout &
NGROK_PID=$!

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Shutting down..."
    kill $SERVER_PID 2>/dev/null
    kill $NGROK_PID 2>/dev/null
    exit 0
}

# Trap cleanup on script exit
trap cleanup SIGINT SIGTERM

echo ""
echo "✅ Both services started!"
echo "📱 Local access: http://localhost:3000"
echo "🌍 Check ngrok dashboard at: http://localhost:4040"
echo ""
echo "Press Ctrl+C to stop both services"

# Wait for processes
wait