const connection = document.querySelector('#live-connection');
const conversation = document.querySelector('#conversation');
const empty = document.querySelector('#live-empty');
const events = new Map();
let socket;
let closed = false;

function receive(event) {
  if (![1, 2].includes(event.mic_id)) return;
  const key = event.event_id || `${event.mic_id}:${event.type}:${event.timestamp}:${event.text}`;
  const previous = events.get(key);
  if (previous?.type === 'transcript.final' && event.type === 'transcript.partial') return;
  events.set(key, event);
}

function render() {
  const all = Array.from(events.values());
  const transcripts = all.filter(event => event.type.startsWith('transcript.'));
  transcripts.sort((a, b) => Date.parse(a.timestamp) - Date.parse(b.timestamp));
  const analyses = all.filter(event => ['reasoning', 'error'].includes(event.type));
  const rows = transcripts.map(event => {
    const row = document.createElement('article');
    row.className = `conversation-turn mic-${event.mic_id}`;
    const label = document.createElement('h2');
    label.textContent = `Microphone ${event.mic_id} · ${new Date(event.timestamp).toLocaleTimeString()}`;
    const text = document.createElement('p');
    text.className = event.type === 'transcript.partial' ? 'transcript-partial' : '';
    text.textContent = event.text;
    row.append(label, text);
    const matches = analyses.filter(analysis => analysis.source_event_id
      ? analysis.source_event_id === event.event_id
      : analysis.mic_id === event.mic_id && analysis.source_text === event.text);
    const heading = document.createElement('h3');
    heading.textContent = 'Reasoning';
    row.append(heading);
    if (!matches.length) {
      const pending = document.createElement('p');
      pending.textContent = event.type === 'transcript.partial'
        ? 'Listening…' : 'Awaiting analysis';
      row.append(pending);
    }
    for (const analysis of matches) {
      const reasoning = document.createElement('p');
      reasoning.textContent = `${new Date(analysis.timestamp).toLocaleTimeString()} · ${analysis.text}`;
      row.append(reasoning);
    }
    return row;
  });
  conversation.replaceChildren(...rows);
  empty.hidden = rows.length > 0;
}

async function refresh() {
  try {
    const response = await fetch('/api/live', { cache: 'no-store' });
    if (!response.ok) throw new Error('Feed unavailable');
    for (const event of await response.json()) receive(event);
    render();
    if (socket?.readyState !== WebSocket.OPEN) {
      connection.textContent = 'Connected — refreshing conversation every 3 seconds';
    }
  } catch {
    if (socket?.readyState !== WebSocket.OPEN) {
      connection.textContent = 'Connection unavailable — retrying automatically';
    }
  }
}

function connect() {
  if (closed) return;
  socket = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws/mic/0`);
  socket.onopen = () => { connection.textContent = 'Live — both microphones connected'; };
  socket.onmessage = message => {
    try { receive(JSON.parse(message.data)); render(); } catch { /* Ignore invalid messages. */ }
  };
  socket.onerror = () => { connection.textContent = 'Reconnecting — conversation polling remains active'; };
  socket.onclose = () => {
    if (!closed) { void refresh(); setTimeout(connect, 2000); }
  };
}

void refresh();
connect();
const poll = setInterval(refresh, 3000);
addEventListener('pagehide', () => {
  closed = true;
  clearInterval(poll);
  socket?.close();
});
