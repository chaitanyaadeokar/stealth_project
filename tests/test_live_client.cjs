const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const code = fs.readFileSync(path.join(__dirname, '../backend/app/static/js/live-client.js'), 'utf8');

test('timeline sorts microphones by timestamp and attaches delayed reasoning without duplicates', async () => {
  const elements = {};
  const make = () => ({ children: [], append(...rows) { this.children.push(...rows); }, replaceChildren(...rows) { this.children = rows; } });
  const element = key => elements[key] ||= make();
  let ws;
  let poll;
  const earlier = { event_id: 'one', mic_id: 1, type: 'transcript.partial', timestamp: '2026-11-21T10:00:00Z', text: 'First draft' };
  const later = { event_id: 'two', mic_id: 2, type: 'transcript.final', timestamp: '2026-11-21T10:00:02Z', text: 'Second speaker' };
  vm.runInNewContext(code, {
    document: { querySelector: element, createElement: make },
    location: { protocol: 'https:', host: 'example.ngrok-free.app' },
    WebSocket: class {
      static OPEN = 1;
      constructor() { ws = this; this.readyState = 1; }
    },
    fetch: async () => ({ ok: true, json: async () => [later, earlier] }),
    setInterval: callback => { poll = callback; return 1; },
    clearInterval() {}, setTimeout() {}, addEventListener() {},
  });
  await new Promise(resolve => setImmediate(resolve));
  const rows = () => element('#conversation').children;
  assert.equal(rows().length, 2);
  assert.match(rows()[0].children[0].textContent, /Microphone 1/);
  assert.match(rows()[1].children[0].textContent, /Microphone 2/);
  const send = event => ws.onmessage({ data: JSON.stringify(event) });
  send({ ...earlier, type: 'transcript.final', text: 'First sentence.' });
  send({ event_id: 'reason', mic_id: 1, type: 'reasoning', source_event_id: 'one', timestamp: '2026-11-21T10:00:10Z', text: 'First speaker analysis' });
  assert.equal(rows().length, 2);
  assert.match(rows()[0].children.at(-1).textContent, /First speaker analysis/);
  assert.equal(rows()[0].children[1].textContent, 'First sentence.');
  await poll();
  assert.equal(rows().length, 2);
  assert.equal(rows()[0].children[1].textContent, 'First sentence.');
  ws.readyState = 3;
  await poll();
  assert.match(element('#live-connection').textContent, /refreshing conversation/);
  assert.equal(element('#live-empty').hidden, true);
});
