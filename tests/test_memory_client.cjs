const { test } = require('node:test');
const assert = require('node:assert/strict');
const vm = require('node:vm');
const fs = require('node:fs');
const path = require('node:path');

test('memory inspector renders committed turns, fallback source and promoted long-term records', async () => {
  const elements = {};
  const get = id => elements[id] ||= { addEventListener() {}, replaceChildren(...rows) { this.children = rows; } };
  let fail = false;
  let refresh;
  vm.runInNewContext(fs.readFileSync(path.join(__dirname, '../backend/app/static/js/memory-client.js'), 'utf8'), {
    document: { getElementById: get, createElement: () => ({}) },
    fetch: async () => {
      if (fail) throw new Error('offline');
      return { ok: true, json: async () => ({
        session_id: 'live-session', window_limit: 20, ttl_seconds: 3600, max_bytes: 900000,
        window: { source: 'postgresql', cache_status: 'miss', data: [{ mic_id: 2, text: '<script>safe text</script>', started_at: '2026-11-21T10:00:00Z' }] },
        state: { source: 'memcached', cache_status: 'hit', data: { current_topic: 'Education' } },
        long_term_status: 'available',
        long_term_message: 'PostgreSQL long-term memory',
        long_term_records: [{ kind: 'evidence', claim: 'A claim', fact_status: 'uncertain', sources: [] }],
      }) };
    },
    AbortSignal,
    setInterval: callback => { refresh = callback; return 1; },
    clearInterval() {}, addEventListener() {},
  });
  await new Promise(resolve => setImmediate(resolve));
  assert.match(get('memory-window-source').textContent, /postgresql.*miss/);
  assert.match(get('memory-turns').children[0].textContent, /Microphone 2.*<script>/);
  assert.match(get('memory-state').textContent, /Education/);
  assert.match(get('memory-long-term').textContent, /PostgreSQL/);
  assert.match(get('memory-long-records').textContent, /evidence/);
  assert.match(get('memory-long-records').textContent, /uncertain/);
  fail = true;
  await refresh();
  assert.match(get('memory-status').textContent, /may be stale/);
  assert.equal(get('memory-refresh').disabled, false);
});
