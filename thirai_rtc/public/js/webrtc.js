class WebRTCManager {
    constructor() {
        this.socket = io();
        this.localStream = null;
        this.peers = new Map();
        this.roomId = null;
        this.userName = null;
        this.currentBandwidth = 1000;
        this.srEnabled = false;
        
        this.initializeSocketEvents();
    }

    async initialize() {
        try {
            await this.getUserMedia();
            this.setupLocalVideo();
            return true;
        } catch (error) {
            console.error('Failed to initialize WebRTC:', error);
            return false;
        }
    }

    async getUserMedia() {
        const constraints = {
            video: {
                width: { ideal: 640 },
                height: { ideal: 480 },
                frameRate: { ideal: 30 }
            },
            audio: {
                echoCancellation: true,
                noiseSuppression: true,
                autoGainControl: true
            }
        };

        this.localStream = await navigator.mediaDevices.getUserMedia(constraints);
    }

    setupLocalVideo() {
        const localVideo = document.getElementById('local-video');
        localVideo.srcObject = this.localStream;
        
        localVideo.onloadedmetadata = () => {
            this.setupVideoProcessing();
        };
    }

    setupVideoProcessing() {
        const video = document.getElementById('local-video');
        const canvas = document.getElementById('local-canvas');
        const ctx = canvas.getContext('2d');
        
        canvas.width = video.videoWidth || 640;
        canvas.height = video.videoHeight || 480;

        this.processVideoFrame(video, canvas, ctx);
    }

    processVideoFrame(video, canvas, ctx) {
        const process = () => {
            if (video.videoWidth === 0) {
                requestAnimationFrame(process);
                return;
            }

            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            
            if (this.srEnabled) {
                this.applySuperResolution(ctx, canvas);
            }

            requestAnimationFrame(process);
        };
        
        process();
    }

    applySuperResolution(ctx, canvas) {
        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        
        window.dispatchEvent(new CustomEvent('videoFrame', {
            detail: {
                imageData: imageData,
                canvas: canvas,
                context: ctx
            }
        }));
    }

    getProcessedStream() {
        if (this.srEnabled) {
            const canvas = document.getElementById('local-canvas');
            return canvas.captureStream(30);
        }
        return this.localStream;
    }

    initializeSocketEvents() {
        this.socket.on('user-joined', (data) => {
            console.log('User joined:', data);
            this.createPeerConnection(data.socketId, true);
        });

        this.socket.on('room-users', (users) => {
            users.forEach(user => {
                this.createPeerConnection(user.socketId, false);
            });
        });

        this.socket.on('offer', async (data) => {
            await this.handleOffer(data.offer, data.sender);
        });

        this.socket.on('answer', async (data) => {
            await this.handleAnswer(data.answer, data.sender);
        });

        this.socket.on('ice-candidate', (data) => {
            this.handleIceCandidate(data.candidate, data.sender);
        });

        this.socket.on('user-left', (socketId) => {
            this.removePeer(socketId);
        });
    }

    createPeerConnection(remoteSocketId, isInitiator) {
        const configuration = {
            iceServers: [
                { urls: 'stun:stun.l.google.com:19302' },
                { urls: 'stun:stun1.l.google.com:19302' }
            ]
        };

        const peerConnection = new RTCPeerConnection(configuration);
        this.peers.set(remoteSocketId, peerConnection);

        this.addLocalStreamToPeer(peerConnection);
        this.setupPeerEventHandlers(peerConnection, remoteSocketId, isInitiator);

        if (isInitiator) {
            this.createOffer(peerConnection, remoteSocketId);
        }
    }

    addLocalStreamToPeer(peerConnection) {
        const stream = this.getProcessedStream();
        stream.getTracks().forEach(track => {
            peerConnection.addTrack(track, stream);
        });
    }

    setupPeerEventHandlers(peerConnection, remoteSocketId, isInitiator) {
        peerConnection.onicecandidate = (event) => {
            if (event.candidate) {
                this.socket.emit('ice-candidate', {
                    candidate: event.candidate,
                    target: remoteSocketId
                });
            }
        };

        peerConnection.ontrack = (event) => {
            this.addRemoteVideo(event.streams[0], remoteSocketId);
        };

        peerConnection.onconnectionstatechange = () => {
            console.log('Connection state:', peerConnection.connectionState);
            this.updateConnectionStatus(peerConnection.connectionState);
        };

        this.configureBandwidth(peerConnection);
    }

    async configureBandwidth(peerConnection) {
        const sender = peerConnection.getSenders().find(s => 
            s.track && s.track.kind === 'video'
        );

        if (sender) {
            try {
                const params = sender.getParameters();
                if (!params.encodings || params.encodings.length === 0) {
                    params.encodings = [{}];
                }
                
                // Set multiple encoding parameters for better throttling
                params.encodings[0].maxBitrate = this.currentBandwidth * 1000;
                params.encodings[0].maxFramerate = this.currentBandwidth < 500 ? 15 : 30;
                
                await sender.setParameters(params);
                console.log(`Bandwidth set to ${this.currentBandwidth} kbps`);
            } catch (error) {
                console.error('Error setting bandwidth:', error);
            }
        }
    }

    async createOffer(peerConnection, remoteSocketId) {
        try {
            const offer = await peerConnection.createOffer();
            await peerConnection.setLocalDescription(offer);
            
            this.socket.emit('offer', {
                offer: offer,
                target: remoteSocketId
            });
        } catch (error) {
            console.error('Error creating offer:', error);
        }
    }

    async handleOffer(offer, remoteSocketId) {
        try {
            const peerConnection = this.peers.get(remoteSocketId);
            await peerConnection.setRemoteDescription(offer);
            
            const answer = await peerConnection.createAnswer();
            await peerConnection.setLocalDescription(answer);
            
            this.socket.emit('answer', {
                answer: answer,
                target: remoteSocketId
            });
        } catch (error) {
            console.error('Error handling offer:', error);
        }
    }

    async handleAnswer(answer, remoteSocketId) {
        try {
            const peerConnection = this.peers.get(remoteSocketId);
            await peerConnection.setRemoteDescription(answer);
        } catch (error) {
            console.error('Error handling answer:', error);
        }
    }

    handleIceCandidate(candidate, remoteSocketId) {
        const peerConnection = this.peers.get(remoteSocketId);
        if (peerConnection) {
            peerConnection.addIceCandidate(candidate).catch(console.error);
        }
    }

    addRemoteVideo(stream, socketId) {
        const remoteVideos = document.getElementById('remote-videos');
        
        let videoWrapper = document.getElementById(`remote-${socketId}`);
        if (!videoWrapper) {
            videoWrapper = document.createElement('div');
            videoWrapper.id = `remote-${socketId}`;
            videoWrapper.className = 'remote-video-wrapper';
            
            const video = document.createElement('video');
            video.autoplay = true;
            video.playsinline = true;
            video.srcObject = stream;
            
            const label = document.createElement('div');
            label.className = 'video-label';
            label.textContent = `User ${socketId.substr(0, 8)}`;
            
            videoWrapper.appendChild(video);
            videoWrapper.appendChild(label);
            remoteVideos.appendChild(videoWrapper);
        }
    }

    removePeer(socketId) {
        const peerConnection = this.peers.get(socketId);
        if (peerConnection) {
            peerConnection.close();
            this.peers.delete(socketId);
        }

        const videoElement = document.getElementById(`remote-${socketId}`);
        if (videoElement) {
            videoElement.remove();
        }
    }

    joinRoom(roomId, userName) {
        this.roomId = roomId;
        this.userName = userName;
        this.socket.emit('join-room', { roomId, userName });
    }

    toggleSR(enabled) {
        this.srEnabled = enabled;
        console.log('Super-resolution:', enabled ? 'enabled' : 'disabled');
        
        this.peers.forEach((peerConnection) => {
            this.updatePeerStream(peerConnection);
        });
    }

    updatePeerStream(peerConnection) {
        const newStream = this.getProcessedStream();
        const sender = peerConnection.getSenders().find(s => 
            s.track && s.track.kind === 'video'
        );

        if (sender) {
            const newTrack = newStream.getVideoTracks()[0];
            sender.replaceTrack(newTrack).catch(console.error);
        }
    }

    updateBandwidth(bandwidth) {
        this.currentBandwidth = bandwidth;
        console.log('Bandwidth updated to:', bandwidth, 'kbps');
        
        this.peers.forEach(async (peerConnection) => {
            await this.configureBandwidth(peerConnection);
        });
        
        // Also update local stream constraints for immediate effect
        this.updateLocalStreamConstraints(bandwidth);
    }

    updateLocalStreamConstraints(bandwidth) {
        const videoTrack = this.localStream.getVideoTracks()[0];
        if (videoTrack && videoTrack.applyConstraints) {
            const constraints = {
                width: bandwidth < 300 ? 320 : bandwidth < 500 ? 480 : 640,
                height: bandwidth < 300 ? 240 : bandwidth < 500 ? 360 : 480,
                frameRate: bandwidth < 500 ? 15 : 30
            };
            
            videoTrack.applyConstraints(constraints).catch(console.error);
        }
    }

    toggleVideo() {
        const videoTrack = this.localStream.getVideoTracks()[0];
        if (videoTrack) {
            videoTrack.enabled = !videoTrack.enabled;
            return videoTrack.enabled;
        }
        return false;
    }

    toggleAudio() {
        const audioTrack = this.localStream.getAudioTracks()[0];
        if (audioTrack) {
            audioTrack.enabled = !audioTrack.enabled;
            return audioTrack.enabled;
        }
        return false;
    }

    updateConnectionStatus(state) {
        const statusElement = document.getElementById('connection-status');
        const statusContainer = statusElement.parentElement;
        
        statusContainer.className = 'status';
        
        switch (state) {
            case 'connected':
                statusElement.textContent = 'Connected';
                statusContainer.classList.add('connected');
                break;
            case 'connecting':
                statusElement.textContent = 'Connecting...';
                statusContainer.classList.add('connecting');
                break;
            case 'disconnected':
            case 'failed':
                statusElement.textContent = 'Connection failed';
                statusContainer.classList.add('error');
                break;
            default:
                statusElement.textContent = 'Connecting...';
                statusContainer.classList.add('connecting');
        }
    }

    leaveRoom() {
        this.peers.forEach((peerConnection, socketId) => {
            this.removePeer(socketId);
        });
        
        if (this.localStream) {
            this.localStream.getTracks().forEach(track => track.stop());
        }
        
        this.socket.disconnect();
    }
}