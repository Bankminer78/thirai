#!/bin/bash

# Build script for Image Roundtrip Mac App

APP_NAME="ImageRoundtrip"
BUILD_DIR="build"
APP_DIR="$BUILD_DIR/$APP_NAME.app"
CONTENTS_DIR="$APP_DIR/Contents"
MACOS_DIR="$CONTENTS_DIR/MacOS"

echo "Building $APP_NAME..."

# Create app bundle structure
mkdir -p "$MACOS_DIR"

# Copy Info.plist
cp Info.plist "$CONTENTS_DIR/"

# Compile Swift app
echo "Compiling Swift code..."
swiftc -o "$MACOS_DIR/$APP_NAME" \
    -target arm64-apple-macos12.0 \
    ImageRoundtripApp.swift

if [ $? -eq 0 ]; then
    echo "✅ Build successful!"
    echo "App bundle created at: $APP_DIR"
    echo ""
    echo "To run the app:"
    echo "open $APP_DIR"
    echo ""
    echo "Or double-click the app bundle in Finder"
else
    echo "❌ Build failed!"
    exit 1
fi