const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const code = fs.readFileSync(
  path.join(__dirname, '../backend/app/static/js/live-client.js'),
  'utf8',
);

test('combined view keeps one revisable row per microphone', () => {
  const transcript = { children: [], prepend(row) { this.children.unshift(row); } };
  class WebSocket {
    constructor(url) { this.url = url; instance = this; }
  }
  let instance;
  vm.runInNewContext(code, {
    document: {
      querySelector: () => transcript,
      createElement: () => ({ className: '', textContent: '' }),
    },
    location: { protocol: 'http:', host: 'localhost:8000' },
    WebSocket,
    JSON,
  });

  const send = (event) => instance.onmessage({ data: JSON.stringify(event) });
  send({ type: 'transcript.partial', mic_id: 1, speaker: 'Mic 1', text: 'Hello' });
  send({ type: 'transcript.partial', mic_id: 2, speaker: 'Mic 2', text: 'Good' });
  send({ type: 'transcript.partial', mic_id: 1, speaker: 'Mic 1', text: 'Hello there' });
  assert.equal(transcript.children.length, 2);
  assert.ok(transcript.children.some((row) => row.textContent === 'Mic 1: Hello there'));
  assert.ok(transcript.children.some((row) => row.textContent === 'Mic 2: Good'));

  send({ type: 'transcript.final', mic_id: 1, speaker: 'Mic 1', text: 'Hello there.' });
  assert.equal(transcript.children.length, 2);
  const final = transcript.children.find((row) => row.textContent === 'Mic 1: Hello there.');
  assert.equal(final.className, '');
});
