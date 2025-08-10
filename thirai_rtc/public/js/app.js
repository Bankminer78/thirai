let webrtcManager = null;
let currentRoomId = null;
let currentUserName = null;

function generateRoomId() {
    return Math.random().toString(36).substr(2, 9);
}

function switchToCallPage() {
    document.getElementById('landing-page').classList.add('hidden');
    document.getElementById('call-page').classList.remove('hidden');
    
    document.getElementById('current-user').textContent = currentUserName;
    document.getElementById('current-room').textContent = currentRoomId;
}

function switchToLandingPage() {
    document.getElementById('call-page').classList.add('hidden');
    document.getElementById('landing-page').classList.remove('hidden');
}

async function joinCall() {
    const nameInput = document.getElementById('user-name');
    const userName = nameInput.value.trim();
    
    if (!userName) {
        alert('Please enter your name');
        return;
    }

    currentUserName = userName;
    currentRoomId = 'main-room'; // Everyone joins the same room
    
    // Update URL to show the main room
    const newUrl = `${window.location.pathname}?room=${currentRoomId}`;
    window.history.pushState({}, '', newUrl);
    
    webrtcManager = new WebRTCManager();
    
    const initialized = await webrtcManager.initialize();
    if (!initialized) {
        alert('Failed to access camera/microphone');
        return;
    }
    
    switchToCallPage();
    webrtcManager.joinRoom(currentRoomId, userName);
    
    setupControls();
}

function setupControls() {
    const srToggle = document.getElementById('sr-toggle');
    const bandwidthSlider = document.getElementById('bandwidth-throttle');
    const bandwidthValue = document.getElementById('bandwidth-value');
    
    srToggle.addEventListener('change', (e) => {
        webrtcManager.toggleSR(e.target.checked);
        
        const localVideo = document.getElementById('local-video');
        const localCanvas = document.getElementById('local-canvas');
        
        if (e.target.checked) {
            localVideo.classList.add('hidden');
            localCanvas.classList.remove('hidden');
        } else {
            localCanvas.classList.add('hidden');
            localVideo.classList.remove('hidden');
        }
    });
    
    bandwidthSlider.addEventListener('input', (e) => {
        const bandwidth = parseInt(e.target.value);
        bandwidthValue.textContent = `${bandwidth} kbps`;
        webrtcManager.updateBandwidth(bandwidth);
    });
    
    window.addEventListener('videoFrame', (event) => {
        handleVideoFrameForSR(event.detail);
    });
}

function handleVideoFrameForSR(frameData) {
    console.log('Video frame available for SR processing:', frameData);
}

function toggleVideo() {
    const enabled = webrtcManager.toggleVideo();
    const button = document.getElementById('toggle-video');
    button.textContent = enabled ? '📹 Video' : '📹 Video (Off)';
    button.style.opacity = enabled ? '1' : '0.5';
}

function toggleAudio() {
    const enabled = webrtcManager.toggleAudio();
    const button = document.getElementById('toggle-audio');
    button.textContent = enabled ? '🎤 Audio' : '🎤 Audio (Off)';
    button.style.opacity = enabled ? '1' : '0.5';
}

function leaveCall() {
    if (webrtcManager) {
        webrtcManager.leaveRoom();
    }
    
    switchToLandingPage();
    
    const newUrl = window.location.pathname;
    window.history.pushState({}, '', newUrl);
    
    document.getElementById('user-name').value = '';
    currentUserName = null;
    currentRoomId = null;
    webrtcManager = null;
}

document.addEventListener('DOMContentLoaded', () => {
    // Always show main-room
    document.getElementById('room-id').textContent = 'main-room';
    
    document.getElementById('user-name').addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            joinCall();
        }
    });
});

window.addEventListener('beforeunload', () => {
    if (webrtcManager) {
        webrtcManager.leaveRoom();
    }
});