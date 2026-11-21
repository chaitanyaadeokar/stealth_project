const { test } = require('node:test');
const assert = require('node:assert/strict');
const vm = require('node:vm');
const fs = require('node:fs');
const path = require('node:path');
const code = fs.readFileSync(path.join(__dirname, '../backend/app/static/js/mic-client.js'), 'utf8');

function setup({ denied = false, failedStart = false, socketFails = false } = {}) {
  const elements = {};
  const calls = [], tracks = [], contexts = [], sockets = [], constraints = [];
  class Element {
    constructor() { this.listeners = {}; this.value = ''; this.disabled = false; this.children = []; this.attrs = {}; this.dataset = { mic: '2' }; }
    addEventListener(name, handler) { (this.listeners[name] ||= []).push(handler); }
    async fire(name, extra = {}) {
      const event = { button: 0, pointerId: 1, preventDefault() {}, target: this, ...extra };
      for (const handler of this.listeners[name] || []) await handler(event);
    }
    setAttribute(key, value) { this.attrs[key] = value; }
    setPointerCapture() {}
    replaceChildren(...children) { this.children = children; }
    add(child) { this.children.push(child); }
    prepend(child) { this.children.unshift(child); }
  }
  const get = (id) => elements[id] ||= new Element();
  const window = new Element();
  class AudioContext {
    constructor() { this.sampleRate = 48000; this.state = 'suspended'; contexts.push(this); }
    async resume() { this.state = 'running'; }
    createMediaStreamSource() { return { connect() {}, disconnect() {} }; }
    createScriptProcessor() { this.processor = { connect() {}, disconnect() {} }; return this.processor; }
    async close() { this.state = 'closed'; }
  }
  class WebSocket {
    static OPEN = 1;
    constructor(url) {
      this.url = url; this.readyState = 0; this.sent = []; sockets.push(this);
      queueMicrotask(() => {
        if (socketFails) this.onerror();
        else { this.readyState = 1; this.onopen(); }
      });
    }
    send(data) {
      this.sent.push(data);
      if (typeof data === 'string' && JSON.parse(data).action === 'flush') {
        const request = JSON.parse(data);
        queueMicrotask(() => this.onmessage({ data: JSON.stringify({ type: 'audio.flushed', request_id: request.request_id }) }));
      }
    }
    close() { this.readyState = 3; this.onclose?.(); }
  }
  vm.runInNewContext(code, {
    document: { querySelector: get, createElement: () => new Element() },
    navigator: { mediaDevices: {
      async getUserMedia(options) {
        constraints.push(options);
        if (denied) throw new Error('Microphone permission denied');
        const track = { stopped: false, stop() { this.stopped = true; } };
        tracks.push(track);
        return { getTracks: () => [track] };
      },
      async enumerateDevices() { return [{ kind: 'audioinput', label: 'USB microphone', deviceId: 'usb' }]; },
    } },
    AudioContext, WebSocket,
    location: { protocol: 'http:', host: 'localhost:8000' },
    fetch: async (url, options) => { calls.push({ url, options }); return { ok: !(failedStart && url.endsWith('/start')), status: 500 }; },
    Option: function(label, value) { this.label = label; this.value = value; },
    addEventListener: window.addEventListener.bind(window),
    setTimeout, clearTimeout, Float32Array, Date, console,
  });
  function audio(sampleCount = 9600) {
    const audio = new Float32Array(sampleCount).fill(0.25);
    contexts.at(-1).processor.onaudioprocess({ inputBuffer: { getChannelData: () => audio } });
  }
  return { get, window, calls, tracks, contexts, sockets, constraints, audio };
}

test('initializes module without currentScript and starts the selected backend microphone', async () => {
  const app = setup();
  assert.equal(app.get('#stop').disabled, true);
  await app.get('#start').fire('click');
  assert.equal(app.get('#connection').textContent, 'Live');
  assert.equal(app.get('#start').disabled, true);
  assert.equal(app.sockets[0].url, 'ws://localhost:8000/ws/mic/2');
  assert.equal(app.calls[0].url, '/api/mics/2/start');
  assert.equal(app.calls[0].options.method, 'POST');
  app.audio();
  assert.equal(app.sockets[0].sent[0].byteLength, 3200 * 4);
  await app.get('#stop').fire('click');
  assert.equal(app.calls.at(-1).url, '/api/mics/2/stop');
  assert.equal(app.tracks[0].stopped, true);
  assert.equal(app.contexts[0].state, 'closed');
  assert.equal(app.sockets[0].readyState, 1);
});

test('mute and push to talk gate audio, including release and focus loss', async () => {
  const app = setup();
  await app.get('#start').fire('click');
  const sent = app.sockets[0].sent;
  const audioCount = () => sent.filter((value) => typeof value !== 'string').length;
  await app.get('#mute').fire('click');
  app.audio();
  assert.equal(audioCount(), 0);
  assert.equal(app.get('#mute').textContent, 'Unmute');
  await app.get('#mute').fire('click');
  await app.get('#mode').fire('click');
  app.audio();
  assert.equal(audioCount(), 0);
  await app.get('#ptt').fire('pointerdown');
  app.audio();
  assert.equal(audioCount(), 1);
  await app.get('#ptt').fire('pointercancel');
  app.audio();
  assert.equal(audioCount(), 1);
  await app.window.fire('keydown', { code: 'Space', target: { tagName: 'BODY' } });
  app.audio();
  assert.equal(audioCount(), 2);
  await app.window.fire('blur');
  app.audio();
  assert.equal(audioCount(), 2);
  await app.window.fire('keydown', { code: 'Space', target: { tagName: 'SELECT' } });
  app.audio();
  assert.equal(audioCount(), 2);
  await app.get('#stop').fire('click');
});

test('stop sends the remaining audio before requesting a server flush', async () => {
  const app = setup();
  await app.get('#start').fire('click');
  app.audio(4800);
  assert.equal(app.sockets[0].sent.length, 0);

  await app.get('#stop').fire('click');

  const sent = app.sockets[0].sent;
  assert.equal(sent[0].byteLength, 1600 * 4);
  assert.equal(JSON.parse(sent[1]).action, 'flush');
  assert.equal(app.calls.at(-1).url, '/api/mics/2/stop');
});

test('device selection replaces the stream and stops previous audio resources', async () => {
  const app = setup();
  await app.get('#start').fire('click');
  app.get('#device').value = 'usb';
  await app.get('#device').fire('change');
  assert.equal(app.tracks[0].stopped, true);
  assert.equal(app.contexts[0].state, 'closed');
  assert.equal(app.constraints[1].audio.deviceId.exact, 'usb');
  assert.equal(app.calls.map(call => call.url).join(','), '/api/mics/2/start,/api/mics/2/stop,/api/mics/2/start');
  await app.get('#stop').fire('click');
});

for (const [name, options] of [
  ['permission denial', { denied: true }],
  ['backend failure', { failedStart: true }],
  ['socket failure', { socketFails: true }],
]) {
  test(name + ' shows an error and allows retry without leaking audio', async () => {
    const app = setup(options);
    await app.get('#start').fire('click');
    assert.equal(app.get('#error').hidden, false);
    assert.equal(app.get('#connection').textContent, 'Unavailable');
    assert.equal(app.get('#start').disabled, false);
    assert.ok(app.tracks.every(track => track.stopped));
    assert.ok(app.contexts.every(context => context.state === 'closed'));
  });
}

test('server events render transcripts and errors in the appropriate areas', async () => {
  const app = setup();
  await app.get('#start').fire('click');
  app.sockets[0].onmessage({ data: JSON.stringify({ type: 'transcript.final', speaker: 'Mic 2', text: 'Hello', timestamp: new Date().toISOString() }) });
  assert.match(app.get('#transcript').children[0].textContent, /Hello/);
  app.sockets[0].onmessage({ data: JSON.stringify({ type: 'error', text: 'Reasoning unavailable' }) });
  assert.equal(app.get('#error').textContent, 'Reasoning unavailable');
  await app.get('#stop').fire('click');
});

test('partial transcript revisions update one row and finalization reuses it', async () => {
  const app = setup();
  await app.get('#start').fire('click');
  const timestamp = new Date().toISOString();
  app.sockets[0].onmessage({ data: JSON.stringify({ type: 'transcript.partial', speaker: 'Mic 2', text: 'The person I am', timestamp }) });
  const row = app.get('#transcript').children[0];
  assert.equal(app.get('#transcript').children.length, 1);
  assert.match(row.textContent, /The person I am/);
  assert.equal(row.className, 'transcript-partial');
  app.sockets[0].onmessage({ data: JSON.stringify({ type: 'transcript.partial', speaker: 'Mic 2', text: 'The person I am working with', timestamp }) });
  assert.equal(app.get('#transcript').children.length, 1);
  assert.match(row.textContent, /working with/);
  app.sockets[0].onmessage({ data: JSON.stringify({ type: 'transcript.final', speaker: 'Mic 2', text: 'The person I am working with is stupid.', timestamp }) });
  assert.equal(app.get('#transcript').children.length, 1);
  assert.match(row.textContent, /is stupid/);
  assert.equal(row.className, '');
  await app.get('#stop').fire('click');
});

test('unexpected disconnection releases audio and makes Start available', async () => {
  const app = setup();
  await app.get('#start').fire('click');
  app.sockets[0].close();
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(app.tracks[0].stopped, true);
  assert.equal(app.get('#start').disabled, false);
  assert.match(app.get('#error').textContent, /Connection lost/);
  assert.equal(app.calls.at(-1).url, '/api/mics/2/stop');
});
