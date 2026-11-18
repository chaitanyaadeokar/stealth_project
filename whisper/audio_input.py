from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Protocol
import math
import subprocess
import wave

import numpy as np


@dataclass(frozen=True)
class AudioChannel:
    """Mono 16 kHz samples, stable identity and session-relative start seconds."""

    channel_id: str
    samples: np.ndarray
    start: float = 0.0

    def __post_init__(self):
        if not isinstance(self.channel_id, str) or not self.channel_id:
            raise ValueError('channel_id must be a nonempty string')
        if not math.isfinite(self.start) or self.start < 0:
            raise ValueError('start must be finite and nonnegative')
        if self.samples.ndim != 1:
            raise ValueError('Each channel must contain mono audio')


class AudioSource(Protocol):
    def __iter__(self) -> Iterator[AudioChannel]:
        ...


@dataclass(frozen=True)
class WavChannel:
    channel_id: str
    path: Path
    start: float = 0.0


class WavInput:
    """Independent mono 16-bit PCM WAV files sharing a session clock."""

    def __init__(self, channels: Iterable[WavChannel]):
        self.channels = tuple(channels)
        ids = [c.channel_id for c in self.channels]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError('Provide at least one channel with unique channel IDs')

    def __iter__(self):
        for channel in self.channels:
            with wave.open(str(channel.path), 'rb') as wav:
                if wav.getnchannels() != 1 or wav.getsampwidth() != 2:
                    raise ValueError('Use one mono 16-bit PCM WAV per channel')
                rate = wav.getframerate()
                samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype='<i2')
                samples = samples.astype(np.float32) / 32768.0
            if rate != 16000 and samples.size:
                decoded = subprocess.run(
                    ['ffmpeg', '-nostdin', '-v', 'error', '-i', str(channel.path),
                     '-f', 'f32le', '-ac', '1', '-ar', '16000', '-'],
                    check=True, capture_output=True,
                )
                samples = np.frombuffer(decoded.stdout, dtype='<f4').copy()
            yield AudioChannel(channel.channel_id, samples, channel.start)


@dataclass(frozen=True)
class Transcript:
    channel_id: str
    speaker_id: str
    start: float
    end: float
    text: str


def transcribe_channels(source: AudioSource, model, on_transcript=None,
                        text_transform=None, **options):
    """Merge finalized segments without clipping or deduplicating overlaps.

    Source adapters may yield blocks with timestamps on a shared clock. This
    finite-input implementation emits only after exhausting input; continuous
    capture needs a watermark policy before publishing finalized results.
    Callbacks receive the same records as the returned chronological list.
    """
    records = []
    for channel in source:
        segments, _ = model.transcribe(channel.samples, **options)
        for segment in segments:
            text = segment.text.strip()
            if not text:
                continue
            start, end = float(segment.start), float(segment.end)
            if not (math.isfinite(start) and math.isfinite(end) and 0 <= start <= end):
                raise ValueError('Whisper returned invalid segment timestamps')
            records.append(Transcript(
                channel.channel_id, 'speaker:' + channel.channel_id,
                channel.start + start, channel.start + end,
                text_transform(text) if text_transform else text,
            ))
    records.sort(key=lambda r: (r.start, r.channel_id, r.end))
    if on_transcript:
        for record in records:
            on_transcript(record)
    return records
