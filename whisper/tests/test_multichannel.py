"""Run with: python -m unittest discover -s tests -p test_multichannel.py"""
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from audio_input import AudioChannel, WavChannel, WavInput, transcribe_channels


class MultichannelTest(unittest.TestCase):
    def test_two_wav_simulation(self):
        class Model:
            def __init__(self):
                self.calls = []

            def transcribe(self, samples, **options):
                self.calls.append(samples.copy())
                if samples[0] > 0:
                    rows = [(2, 4, 'first channel'), (6, 7, 'later')]
                else:
                    rows = [(0, 1, 'second channel'), (2, 3, 'overlap')]
                return (iter(SimpleNamespace(start=a, end=b, text=t)
                             for a, b, t in rows), None)

        with tempfile.TemporaryDirectory() as directory:
            channels = []
            for name, value in [('left', 1000), ('right', -1000)]:
                path = Path(directory) / (name + '.wav')
                with wave.open(str(path), 'wb') as wav:
                    wav.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
                    wav.writeframes(np.full(128000, value, dtype='<i2').tobytes())
                channels.append(WavChannel(name, path))
            model = Model()
            delivered = []
            records = transcribe_channels(WavInput(channels), model, delivered.append)
            self.assertEqual([r.text for r in records],
                             ['second channel', 'first channel', 'overlap', 'later'])
            self.assertEqual([r.speaker_id for r in records],
                             ['speaker:right', 'speaker:left', 'speaker:right', 'speaker:left'])
            self.assertEqual(delivered, records)
            self.assertLess(records[2].start, records[1].end)
            self.assertEqual(len(model.calls), 2)
            self.assertGreater(model.calls[0][0], 0)
            self.assertLess(model.calls[1][0], 0)
            self.assertEqual(records, transcribe_channels(WavInput(reversed(channels)), Model()))

    def test_offsets_and_transform(self):
        class Model:
            def transcribe(self, samples, **options):
                return iter([SimpleNamespace(start=1, end=2, text=' hello ')]), None
        records = transcribe_channels(
            [AudioChannel('usb:3', np.zeros(16000), 10)], Model(), text_transform=str.upper)
        self.assertEqual((records[0].start, records[0].end), (11, 12))
        self.assertEqual(records[0].text, 'HELLO')
        self.assertEqual(records[0].speaker_id, 'speaker:usb:3')

    def test_wav_resampling_and_stereo_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'input.wav'
            with wave.open(str(path), 'wb') as wav:
                wav.setparams((1, 2, 8000, 0, 'NONE', 'not compressed'))
                wav.writeframes(np.full(8000, 1000, dtype='<i2').tobytes())
            channel = next(iter(WavInput([WavChannel('a', path)])))
            self.assertEqual(channel.samples.shape, (16000,))
            self.assertEqual(channel.samples.dtype, np.float32)
            with wave.open(str(path), 'wb') as wav:
                wav.setparams((2, 2, 16000, 0, 'NONE', 'not compressed'))
                wav.writeframes(np.zeros(32000, dtype='<i2').tobytes())
            with self.assertRaises(ValueError):
                list(WavInput([WavChannel('a', path)]))

    def test_invalid_input(self):
        with self.assertRaises(ValueError):
            WavInput([WavChannel('a', 'a.wav'), WavChannel('a', 'b.wav')])
        with self.assertRaises(ValueError):
            AudioChannel('a', np.zeros((10, 2)))


if __name__ == '__main__':
    unittest.main()
