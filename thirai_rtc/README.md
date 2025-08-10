# Thirai RTC - Video Calling Platform

A WebRTC-based video calling platform with integrated face super-resolution capabilities.

## Features

- **Simple Join Process**: Enter your name and join a video call instantly
- **Face Super-Resolution Toggle**: Enable/disable SR processing on video streams
- **Bandwidth Throttling**: Adjustable bandwidth control (50-2000 kbps)
- **Real-time Communication**: WebRTC peer-to-peer video calls
- **Video Stream Access**: Easy integration point for face enhancement algorithms

## Quick Start

1. **Install dependencies**:
   ```bash
   cd thirai_rtc
   npm install
   ```

2. **Start the server**:
   ```bash
   npm start
   ```

3. **Open your browser** and navigate to `http://localhost:3000`

4. **Join a call**:
   - Enter your name
   - Click "Join Call"
   - Share the room URL with others to join the same call

## Architecture

### Backend (Node.js + Socket.io)
- **server/index.js**: WebRTC signaling server
- Handles room management and peer connection signaling
- Supports multiple concurrent rooms

### Frontend
- **public/index.html**: Landing page and call interface
- **public/js/webrtc.js**: WebRTC connection management
- **public/js/app.js**: UI logic and controls
- **public/css/style.css**: Responsive styling

### Key Components

1. **Landing Page**: Name input and room joining
2. **Video Interface**: Local and remote video streams
3. **Controls Panel**: SR toggle, bandwidth throttling, media controls
4. **Stream Processing**: Canvas-based video frame access for SR integration

## Super-Resolution Integration

The platform provides easy access to video frames for face enhancement:

```javascript
// Listen for video frames when SR is enabled
window.addEventListener('videoFrame', (event) => {
    const { imageData, canvas, context } = event.detail;
    
    // Your face enhancement logic here
    // Process imageData and draw back to canvas
    
    context.putImageData(enhancedImageData, 0, 0);
});
```

### Integration Points

- **Video Frame Access**: Real-time access to video frames via Canvas API
- **Stream Replacement**: Automatic stream switching when SR is toggled
- **Performance Optimized**: 30fps frame processing with requestAnimationFrame

## Bandwidth Control

The bandwidth throttle affects video quality:
- **50-500 kbps**: Low quality, suitable for testing choppy video
- **500-1000 kbps**: Medium quality
- **1000-2000 kbps**: High quality

## Usage

1. **Start a call**: Enter name, click "Join Call"
2. **Enable SR**: Toggle "Enable Face Super-Resolution"
3. **Adjust bandwidth**: Use the slider to control video quality
4. **Invite others**: Share the room URL
5. **Control media**: Toggle video/audio, leave call

## Development

```bash
# Install dependencies
npm install

# Start development server with auto-reload
npm run dev

# Start production server
npm start
```

## Browser Requirements

- Modern browsers with WebRTC support
- Camera and microphone permissions required
- HTTPS recommended for production deployment

## Next Steps for Face Enhancement

1. Integrate your VAE model for face super-resolution
2. Process video frames in the `handleVideoFrameForSR` function
3. Optimize for real-time performance (consider Web Workers)
4. Add face detection to target enhancement areas
5. Implement quality metrics and adaptive processing