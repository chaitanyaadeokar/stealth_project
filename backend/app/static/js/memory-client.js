const memoryElement = id => document.getElementById(id);
let memoryBusy = false;
let memoryClosed = false;

function sourceLabel(part) {
  return `Source: ${part.source} · Cache: ${part.cache_status}${part.message ? ' · ' + part.message : ''}`;
}

function renderMemory(data) {
  memoryElement('memory-session').textContent = `Live session: ${data.session_id}`;
  memoryElement('memory-config').textContent = `Up to ${data.window_limit} turns · Configured TTL: ${data.ttl_seconds}s · Blob limit: ${data.max_bytes} bytes`;
  memoryElement('memory-window-source').textContent = sourceLabel(data.window);
  memoryElement('memory-state-source').textContent = sourceLabel(data.state);
  const turns = Array.isArray(data.window.data) ? data.window.data : [];
  const rows = turns.map(turn => {
    const row = document.createElement('li');
    row.textContent = `Microphone ${turn.mic_id} · ${new Date(turn.started_at).toLocaleTimeString()} — ${turn.text}`;
    return row;
  });
  if (!rows.length) {
    const row = document.createElement('li');
    row.textContent = data.window.source === 'unavailable' ? 'Memory unavailable.' : 'No committed turns in this window yet.';
    rows.push(row);
  }
  memoryElement('memory-turns').replaceChildren(...rows);
  memoryElement('memory-state').textContent = data.state.data === null ? 'State unavailable.' : JSON.stringify(data.state.data, null, 2);
  memoryElement('memory-debug').textContent = JSON.stringify({
    window_key: data.window_key, state_key: data.state_key,
    pending_cache_updates: data.pending_cache_updates,
    pending_database_writes: data.pending_database_writes,
    pending_promotions: data.pending_promotions,
  }, null, 2);
  memoryElement('memory-long-term').textContent = data.long_term_message;
  const records = data.long_term_records || [];
  memoryElement('memory-long-records').textContent = records.length
    ? JSON.stringify(records, null, 2)
    : (data.long_term_status === 'unavailable' ? 'Records unavailable.' : 'No promoted records for this session yet.');
  const unavailable = data.window.source === 'unavailable' || data.state.source === 'unavailable' || data.long_term_status === 'unavailable';
  memoryElement('memory-status').textContent = `${unavailable ? 'Some memory data is unavailable' : 'Updated'} · ${new Date().toLocaleTimeString()}`;
}

async function refreshMemory() {
  if (memoryBusy || memoryClosed) return;
  memoryBusy = true;
  memoryElement('memory-refresh').disabled = true;
  try {
    const response = await fetch('/api/memory', { cache: 'no-store', signal: AbortSignal.timeout(5000) });
    if (!response.ok) throw new Error('Memory endpoint unavailable');
    renderMemory(await response.json());
  } catch {
    memoryElement('memory-status').textContent = 'Could not refresh memory. Displayed data may be stale; retrying automatically.';
  } finally {
    memoryBusy = false;
    memoryElement('memory-refresh').disabled = false;
  }
}

memoryElement('memory-refresh').addEventListener('click', refreshMemory);
void refreshMemory();
const memoryTimer = setInterval(refreshMemory, 5000);
addEventListener('pagehide', () => { memoryClosed = true; clearInterval(memoryTimer); });
