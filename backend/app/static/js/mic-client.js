const mic = document.querySelector('script[data-mic]').dataset.mic;
const $ = (selector) => document.querySelector(selector);
let muted = false, openMic = true, talking = false, running = false, busy = false;
let stream, context, source, processor, socket;
let pipelineStarted = false;
let partialRow;
let resampler;
let pendingNetworkAudio = new Float32Array(0);
let flushRequestId = 0;
const flushRequests = new Map();
let audioSinceFlush = false;
const TARGET_SAMPLE_RATE = 16000;
const NETWORK_FRAME_SAMPLES = 3200;

class StreamingResampler {
  constructor(sourceRate, targetRate) {
    this.ratio = sourceRate / targetRate;
    this.tail = new Float32Array(0);
    this.position = 0;
  }

  push(input) {
    if (!input.length) return new Float32Array(0);
    const samples = new Float32Array(this.tail.length + input.length);
    samples.set(this.tail);
    samples.set(input, this.tail.length);
    const output = [];
    let position = this.position;
    while (position + 1 < samples.length) {
      const index = Math.floor(position);
      const fraction = position - index;
      output.push(
        samples[index] + (samples[index + 1] - samples[index]) * fraction,
      );
      position += this.ratio;
    }
    const keepIndex = Math.min(Math.floor(position), samples.length - 1);
    this.tail = samples.slice(keepIndex);
    this.position = position - keepIndex;
    return Float32Array.from(output);
  }
}

function sendAudio(audio) {
  if (audio.length) audioSinceFlush = true;
  const combined = new Float32Array(pendingNetworkAudio.length + audio.length);
  combined.set(pendingNetworkAudio);
  combined.set(audio, pendingNetworkAudio.length);
  let offset = 0;
  while (
    combined.length - offset >= NETWORK_FRAME_SAMPLES
    && socket?.readyState === WebSocket.OPEN
  ) {
    socket.send(combined.slice(offset, offset + NETWORK_FRAME_SAMPLES).buffer);
    audioSinceFlush = true;
    offset += NETWORK_FRAME_SAMPLES;
  }
  pendingNetworkAudio = combined.slice(offset);
}

function flushPendingAudio() {
  if (pendingNetworkAudio.length && socket?.readyState === WebSocket.OPEN) {
    socket.send(pendingNetworkAudio.buffer);
    audioSinceFlush = true;
  }
  pendingNetworkAudio = new Float32Array(0);
}

function requestAudioFlush() {
  if (socket?.readyState !== WebSocket.OPEN || !audioSinceFlush) {
    return Promise.resolve();
  }
  flushPendingAudio();
  audioSinceFlush = false;
  const requestId = String(++flushRequestId);
  return new Promise((resolve) => {
    const timeout = setTimeout(() => {
      flushRequests.delete(requestId);
      resolve();
    }, 5000);
    flushRequests.set(requestId, () => {
      clearTimeout(timeout);
      resolve();
    });
    socket.send(JSON.stringify({ action: 'flush', request_id: requestId }));
  });
}

function handleAudio(audio) {
  const rms = Math.sqrt(
    audio.reduce((sum, value) => sum + value * value, 0) / audio.length,
  );
  $('#level').value = Math.min(1, rms * 4);
  if (!running || muted || (!openMic && !talking)) return;
  sendAudio(resampler.push(audio));
}

function showError(error) {
  $('#error').textContent = error.message || String(error);
  $('#error').hidden = false;
}
function updateControls() {
  $('#start').disabled = busy || running;
  $('#stop').disabled = busy || !running;
  $('#device').disabled = busy;
  $('#mute').disabled = busy || !running;
  $('#mute').textContent = muted ? 'Unmute' : 'Mute';
  $('#mute').setAttribute('aria-pressed', String(muted));
  $('#mode').textContent = openMic ? 'Open mic' : 'Push to talk';
  $('#mode').setAttribute('aria-pressed', String(!openMic));
  $('#ptt').disabled = busy || !running || openMic || muted;
  $('#ptt').setAttribute('aria-pressed', String(talking));
}
function setTalking(value) {
  const wasTalking = talking;
  talking = value && running && !openMic && !muted;
  if (wasTalking && !talking) void requestAudioFlush();
  updateControls();
}
async function controlPipeline(action, keepalive = false) {
  const response = await fetch('/api/mics/' + mic + '/' + action, { method: 'POST', keepalive });
  if (!response.ok) throw new Error('Could not ' + action + ' microphone (' + response.status + '). Please retry.');
}
function addEvent(event) {
  if (event.type === 'error') return showError(new Error(event.text));
  if (event.type === 'status') { $('#connection').textContent = event.text; return; }
  const row = document.createElement('p');
  const time = new Date(event.timestamp).toLocaleTimeString();
  const source = event.source_text ? 'For “' + event.source_text + '” — ' : '';
  row.textContent = event.speaker + ' · ' + time + ' — ' + source + event.text;
  if (event.type === 'transcript.partial') {
    if (!partialRow) {
      partialRow = row;
      partialRow.className = 'transcript-partial';
      $('#transcript').prepend(partialRow);
    } else {
      partialRow.textContent = row.textContent;
    }
    return;
  }
  if (event.type === 'transcript.final') {
    if (partialRow) {
      partialRow.textContent = row.textContent;
      partialRow.className = '';
      partialRow = undefined;
    } else {
      $('#transcript').prepend(row);
    }
    return;
  }
  $('#reasoning').prepend(row);
}
function connectSocket() {
  if (socket?.readyState === WebSocket.OPEN) return Promise.resolve();
  return new Promise((resolve, reject) => {
    const connection = new WebSocket((location.protocol === 'https:' ? 'wss' : 'ws') + '://' + location.host + '/ws/mic/' + mic);
    socket = connection;
    const timeout = setTimeout(() => {
      reject(new Error('Connection timed out. Please retry.'));
      connection.close();
    }, 10000);
    connection.onopen = () => { clearTimeout(timeout); resolve(); };
    connection.onerror = () => { clearTimeout(timeout); reject(new Error('Could not connect to the server. Please retry.')); };
    connection.onclose = () => {
      clearTimeout(timeout);
      reject(new Error('Connection closed. Please retry.'));
      if (running && socket === connection) {
        void stop().then(() => showError(new Error('Connection lost. Start the microphone to reconnect.')));
      }
    };
    connection.onmessage = (message) => {
      try {
        const event = JSON.parse(message.data);
        if (event.type === 'audio.flushed') {
          const resolve = flushRequests.get(event.request_id);
          flushRequests.delete(event.request_id);
          resolve?.();
          return;
        }
        addEvent(event);
      }
      catch { showError(new Error('Received an unreadable server message.')); }
    };
  });
}
async function releaseAudio(closeSocket = true) {
  running = false;
  talking = false;
  if (closeSocket && socket) { socket.onclose = null; socket.close(); socket = undefined; }
  if (processor) {
    processor.onaudioprocess = null;
    if (processor.port) {
      processor.port.onmessage = null;
      processor.port.close();
    }
    processor.disconnect();
    processor = undefined;
  }
  source?.disconnect();
  source = undefined;
  stream?.getTracks().forEach((track) => track.stop());
  stream = undefined;
  const previousContext = context;
  context = undefined;
  if (previousContext && previousContext.state !== 'closed') await previousContext.close();
  resampler = undefined;
  pendingNetworkAudio = new Float32Array(0);
  audioSinceFlush = false;
  $('#level').value = 0;
}
async function stop() {
  busy = true;
  running = false;
  updateControls();
  try {
    await requestAudioFlush();
    if (pipelineStarted) { await controlPipeline('stop'); pipelineStarted = false; }
    await releaseAudio(false);
    $('#connection').textContent = 'Stopped';
  } catch (error) { showError(error); }
  finally { busy = false; updateControls(); }
}
async function start() {
  if (busy || running) return;
  busy = true;
  $('#error').hidden = true;
  $('#connection').textContent = 'Connecting';
  updateControls();
  try {
    if (!navigator.mediaDevices?.getUserMedia) throw new Error('Microphone access requires HTTPS or localhost and a supported browser.');
    // Resume during the button gesture so browsers allow audio processing.
    context = new AudioContext({ sampleRate: TARGET_SAMPLE_RATE, latencyHint: 'interactive' });
    await context.resume();
    const deviceId = $('#device').value;
    stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        ...(deviceId ? { deviceId: { exact: deviceId } } : {}),
        channelCount: { ideal: 1 },
        echoCancellation: { ideal: true },
        noiseSuppression: { ideal: true },
        autoGainControl: { ideal: true },
      },
    });
    source = context.createMediaStreamSource(stream);
    if (context.audioWorklet && typeof AudioWorkletNode !== 'undefined') {
      await context.audioWorklet.addModule('/static/js/pcm-capture.js');
      processor = new AudioWorkletNode(context, 'pcm-capture');
      processor.port.onmessage = (event) => handleAudio(event.data);
    } else {
      processor = context.createScriptProcessor(2048, 1, 1);
      processor.onaudioprocess = (event) => {
        handleAudio(event.inputBuffer.getChannelData(0));
      };
    }
    resampler = new StreamingResampler(context.sampleRate, TARGET_SAMPLE_RATE);
    pendingNetworkAudio = new Float32Array(0);
    audioSinceFlush = false;
    await connectSocket();
    await controlPipeline('start');
    pipelineStarted = true;
    if (socket.readyState !== WebSocket.OPEN) throw new Error('Connection closed. Please retry.');
    source.connect(processor);
    processor.connect(context.destination);
    running = true;
    $('#connection').textContent = 'Live';
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      $('#device').replaceChildren(new Option('Default microphone', ''));
      devices.filter((device) => device.kind === 'audioinput').forEach((device) => $('#device').add(new Option(device.label || 'Microphone', device.deviceId)));
      $('#device').value = deviceId;
    } catch { /* Recording can continue without a device list. */ }
  } catch (error) {
    await stop();
    await releaseAudio();
    $('#connection').textContent = 'Unavailable';
    showError(error);
  } finally { busy = false; updateControls(); }
}
$('#start').addEventListener('click', start);
$('#stop').addEventListener('click', stop);
$('#mute').addEventListener('click', () => {
  muted = !muted;
  setTalking(false);
  if (muted) void requestAudioFlush();
});
$('#mode').addEventListener('click', () => { openMic = !openMic; setTalking(false); });
$('#ptt').addEventListener('pointerdown', (event) => {
  if (event.button !== 0 || $('#ptt').disabled) return;
  event.preventDefault();
  $('#ptt').setPointerCapture(event.pointerId);
  setTalking(true);
});
for (const name of ['pointerup', 'pointercancel', 'lostpointercapture']) $('#ptt').addEventListener(name, () => setTalking(false));
addEventListener('keydown', (event) => {
  const typing = event.target.isContentEditable || ['INPUT', 'TEXTAREA', 'SELECT', 'BUTTON', 'A'].includes(event.target.tagName);
  if (event.code !== 'Space' || !running || openMic || typing) return;
  event.preventDefault();
  setTalking(true);
});
$('#ptt').addEventListener('keydown', (event) => {
  if (!['Space', 'Enter'].includes(event.code)) return;
  event.preventDefault();
  setTalking(true);
});
addEventListener('keyup', (event) => { if (['Space', 'Enter'].includes(event.code)) setTalking(false); });
addEventListener('blur', () => setTalking(false));
$('#device').addEventListener('change', async () => {
  if (!running || busy) return;
  await stop();
  await start();
});
addEventListener('pagehide', () => {
  void releaseAudio(true);
  if (pipelineStarted) void controlPipeline('stop', true).catch(() => {});
});
updateControls();
