"""Optional cached-model smoke test: python tests/simulate_multichannel.py."""
import sys
import subprocess
import tempfile
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from faster_whisper import WhisperModel
from audio_input import WavChannel, WavInput, transcribe_channels


def main():
    # Use only an already cached model; never download for this smoke test.
    root = Path.home() / '.cache' / 'whisper'
    model = WhisperModel('large-v3-turbo', device='cpu', compute_type='int8',
                         download_root=str(root), local_files_only=True, cpu_threads=4)
    decoded = subprocess.run(
        ['ffmpeg', '-nostdin', '-v', 'error', '-i',
         str(Path(__file__).with_name('jfk.flac')), '-f', 'f32le',
         '-ac', '1', '-ar', '16000', '-'], check=True, capture_output=True)
    speech = np.frombuffer(decoded.stdout, dtype='<f4')
    with tempfile.TemporaryDirectory() as directory:
        channels = []
        for name, offset in [('left', 0), ('right', 1)]:
            path = Path(directory) / (name + '.wav')
            with wave.open(str(path), 'wb') as wav:
                wav.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
                wav.writeframes((speech * 32767).astype('<i2').tobytes())
            channels.append(WavChannel(name, path, offset))
        records = transcribe_channels(WavInput(channels), model, language='en',
                                      condition_on_previous_text=False, beam_size=1)
        for speaker in ['speaker:left', 'speaker:right']:
            text = ' '.join(r.text for r in records if r.speaker_id == speaker).lower()
            assert 'ask not' in text and 'country' in text, text
        assert records == sorted(records, key=lambda r: (r.start, r.channel_id, r.end))
        left = [r for r in records if r.channel_id == 'left']
        right = [r for r in records if r.channel_id == 'right']
        assert len(left) == len(right) and left
        for a, b in zip(left, right):
            assert a.text == b.text
            assert abs(b.start - a.start - 1) < 1e-6
        assert any(a.start < b.end and b.start < a.end for a in left for b in right)
        for r in records:
            print(f'{r.start:.2f}-{r.end:.2f} {r.speaker_id}: {r.text}')
        print('Real Whisper two-channel smoke test passed.')


if __name__ == '__main__':
    main()
