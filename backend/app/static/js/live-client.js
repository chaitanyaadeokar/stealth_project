const out = document.querySelector('#transcript');
const ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws/mic/0`);
const partialRows = new Map();
ws.onmessage = (message) => {
  const event = JSON.parse(message.data);
  const row = document.createElement('p');
  row.textContent = `${event.speaker}: ${event.text}`;
  if (event.type === 'transcript.partial') {
    row.className = 'transcript-partial';
    const current = partialRows.get(event.mic_id);
    if (current) current.textContent = row.textContent;
    else { partialRows.set(event.mic_id, row); out.prepend(row); }
  } else if (event.type === 'transcript.final') {
    const current = partialRows.get(event.mic_id);
    if (current) { current.textContent = row.textContent; current.className = ''; partialRows.delete(event.mic_id); }
    else out.prepend(row);
  }
};
