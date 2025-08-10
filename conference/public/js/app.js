let connect, createLocalTracks, RoomEvent, ParticipantEvent;
try {
  ({ connect, createLocalTracks, RoomEvent, ParticipantEvent } = await import('https://cdn.jsdelivr.net/npm/livekit-client@2/dist/livekit-client.esm.min.js'));
  console.log('[diag] livekit-client loaded');
} catch (e) {
  console.error('[diag] Failed to load livekit-client', e);
  alert('Failed to load LiveKit client library. Check network/HTTPS.');
}

const elements = {
  name: document.getElementById('name'),
  join: document.getElementById('join'),
  leave: document.getElementById('leave'),
  status: document.getElementById('status-text'),
  localVideo: document.getElementById('localVideo'),
  remotes: document.getElementById('remotes'),
};

let room = null;

function setStatus(text) { elements.status.textContent = text; }
function setConnectedState(connected) {
  elements.join.disabled = connected;
  elements.leave.disabled = !connected;
}

function attachVideo(track, element) {
  const mediaEl = element || document.createElement('video');
  mediaEl.autoplay = true;
  mediaEl.playsInline = true;
  mediaEl.muted = element ? element.muted : false;
  track.attach(mediaEl);
  return mediaEl;
}

function addRemoteParticipant(participant) {
  const container = document.createElement('div');
  container.className = 'remote-tile';

  const videoEl = document.createElement('video');
  videoEl.autoplay = true;
  videoEl.playsInline = true;
  container.appendChild(videoEl);

  const label = document.createElement('div');
  label.className = 'remote-label';
  label.textContent = participant.name || participant.identity;
  container.appendChild(label);

  elements.remotes.appendChild(container);

  const onTrackSubscribed = (track, pub) => {
    if (track.kind === 'video' || track.kind === 'audio') {
      track.attach(videoEl);
    }
  };
  participant.on(ParticipantEvent.TrackSubscribed, onTrackSubscribed);

  participant.tracks.forEach((pub) => {
    if (pub.isSubscribed && pub.track) {
      onTrackSubscribed(pub.track, pub);
    }
  });

  participant.on(ParticipantEvent.ParticipantDisconnected, () => {
    container.remove();
  });
}

async function joinRoom() {
  setStatus('Requesting token...');
  let data;
  try {
    const resp = await fetch(`/get_token?name=${encodeURIComponent(elements.name.value || '')}`);
    data = await resp.json();
  } catch (e) {
    console.error('[diag] /get_token failed', e);
    setStatus('Failed to reach backend /get_token');
    return;
  }
  if (!data.token || !data.url) {
    console.error('[diag] token response', data);
    setStatus('Failed to obtain token');
    return;
  }

  setStatus('Connecting...');
  try {
    room = await connect(data.url, data.token, {
      autoSubscribe: true,
    });
  } catch (e) {
    console.error('[diag] connect failed', e);
    setStatus(`Connect failed: ${e?.message || e}`);
    return;
  }

  setStatus(`Connected to room: ${data.room}`);
  setConnectedState(true);

  // Publish local tracks
  const localTracks = await createLocalTracks({
    audio: true,
    video: { facingMode: 'user' },
  });

  for (const track of localTracks) {
    await room.localParticipant.publishTrack(track);
    if (track.kind === 'video') {
      track.attach(elements.localVideo);
    }
  }

  // Render existing participants
  room.participants.forEach((p) => addRemoteParticipant(p));

  room.on(RoomEvent.ParticipantConnected, (p) => addRemoteParticipant(p));
  room.on(RoomEvent.ParticipantDisconnected, (p) => {
    // cleanup handled in participant.disconnect listener
  });

  room.on(RoomEvent.Disconnected, () => {
    setConnectedState(false);
    setStatus('Disconnected');
    elements.remotes.innerHTML = '';
    elements.localVideo.srcObject = null;
    room = null;
  });
}

function leaveRoom() {
  if (room) {
    room.disconnect();
  }
}

elements.join.addEventListener('click', () => joinRoom().catch((e) => setStatus(`Error: ${e?.message || e}`)));
elements.leave.addEventListener('click', leaveRoom);

window.addEventListener('beforeunload', leaveRoom);
