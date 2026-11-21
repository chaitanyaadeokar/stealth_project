import asyncio
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from backend.app.core.config import Settings
from backend.app.pipelines.base import Pipeline
from backend.app.pipelines.whisper_pipeline import WhisperPipeline
from backend.app.pipelines.worker import PipelineWorker
from backend.app.schemas.events import LiveEvent


class FakeStreamingPipeline(Pipeline):
    def __init__(self) -> None:
        self.revision = 0
        self.partial = ""
        self.analyzed: list[str] = []
        self.fail_next = False

    async def process(self, chunk: bytes) -> list[LiveEvent]:
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("temporary transcription failure")
        self.revision += 1
        self.partial = (
            "The person I am working"
            if self.revision == 1
            else "The person I am working with is stupid."
        )
        return [
            LiveEvent(type="transcript.partial", mic_id=1, text=self.partial)
        ]

    async def finalize_pending(self) -> list[LiveEvent]:
        if not self.partial:
            return []
        text, self.partial = self.partial, ""
        return [LiveEvent(type="transcript.final", mic_id=1, text=text)]

    async def on_final(self, event: LiveEvent) -> None:
        self.analyzed.append(event.text)


class StreamingPipelineTests(unittest.IsolatedAsyncioTestCase):
    async def test_final_decode_uses_accuracy_settings_and_vocabulary(self) -> None:
        pipeline = WhisperPipeline(1, Settings(
            _env_file=None,
            whisper_final_beam_size=5,
            whisper_initial_prompt="ExampleName, ExamplePlace",
        ))
        pipeline._model = object()
        audio = np.ones(16_000, dtype=np.float32).tobytes()
        with patch(
            "backend.app.pipelines.whisper_pipeline.live_whisper.transcribe_audio",
            return_value=([SimpleNamespace(text="A complete sentence.")], None),
        ) as transcribe:
            await pipeline.process(audio)
            self.assertEqual(transcribe.call_args.kwargs["beam_size"], 1)
            await pipeline.process_final(audio)
            self.assertEqual(transcribe.call_args.kwargs["beam_size"], 5)
            self.assertEqual(
                transcribe.call_args.kwargs["initial_prompt"],
                "ExampleName, ExamplePlace",
            )

    async def test_whisper_revises_one_partial_and_finalizes_once(self) -> None:
        pipeline = WhisperPipeline(1, Settings(_env_file=None))
        revisions = iter(
            [
                [SimpleNamespace(text="The person I am working")],
                [SimpleNamespace(text="The person I am working with is stupid.")],
                [SimpleNamespace(text="The person I am working with is stupid.")],
            ]
        )
        pipeline._transcribe = lambda _audio, _beam_size: next(revisions)
        audio = np.ones(16_000, dtype=np.float32).tobytes()

        first = await pipeline.process(audio)
        second = await pipeline.process(audio)
        unchanged = await pipeline.process(audio)
        final = await pipeline.finalize_pending()

        self.assertEqual([event.type for event in first + second], [
            "transcript.partial",
            "transcript.partial",
        ])
        self.assertEqual(second[0].text, "The person I am working with is stupid.")
        self.assertEqual(unchanged, [])
        self.assertEqual([event.text for event in final], [
            "The person I am working with is stupid."
        ])
        self.assertEqual(await pipeline.finalize_pending(), [])

    async def test_finalized_utterance_is_split_into_sentences(self) -> None:
        pipeline = WhisperPipeline(1, Settings(_env_file=None))
        pipeline._last_partial = "The sky is blue. Grass is green."

        events = await pipeline.finalize_pending()

        self.assertEqual(
            [event.text for event in events],
            ["The sky is blue.", "Grass is green."],
        )

    async def test_reasoning_runs_after_final_without_blocking(self) -> None:
        settings = Settings(
            _env_file=None,
            groq_api_key="test-key",
            llm_timeout_sec=1,
        )
        pipeline = WhisperPipeline(1, settings)
        published: list[LiveEvent] = []

        async def publish(event: LiveEvent) -> None:
            published.append(event)

        pipeline.set_event_publisher(publish)

        def analyze(statement: str) -> str:
            time.sleep(0.05)
            return f"Analysis of {statement}"

        final = LiveEvent(
            type="transcript.final",
            mic_id=1,
            text="The sentence is complete.",
        )
        with patch(
            "backend.app.pipelines.whisper_pipeline.live_whisper.fact_check_statement",
            side_effect=analyze,
        ):
            started = time.monotonic()
            await pipeline.on_final(final)
            self.assertLess(time.monotonic() - started, 0.04)
            await asyncio.gather(*list(pipeline._analysis_tasks))

        self.assertEqual(len(published), 1)
        self.assertEqual(published[0].type, "reasoning")
        self.assertEqual(published[0].source_text, final.text)

    async def test_short_pause_revises_but_sustained_silence_finalizes_once(self) -> None:
        settings = Settings(
            _env_file=None,
            audio_sample_rate=16_000,
            streaming_window_sec=0.35,
            streaming_update_sec=0.2,
            sentence_silence_sec=0.8,
            speech_rms_threshold=0.01,
        )
        pipeline = FakeStreamingPipeline()
        published: list[LiveEvent] = []

        async def publish(event: LiveEvent) -> None:
            published.append(event)

        worker = PipelineWorker(1, pipeline, publish, settings)
        voice = np.full(int(16_000 * 0.4), 0.2, dtype=np.float32).tobytes()
        short_pause = np.zeros(int(16_000 * 0.4), dtype=np.float32).tobytes()
        sentence_pause = np.zeros(int(16_000 * 0.8), dtype=np.float32).tobytes()

        await worker._handle_chunk(voice)
        await worker._handle_chunk(short_pause)
        self.assertFalse(any(event.type == "transcript.final" for event in published))
        await worker._handle_chunk(voice)
        await worker._handle_chunk(sentence_pause)

        finals = [event for event in published if event.type == "transcript.final"]
        self.assertEqual(len(finals), 1)
        self.assertEqual(finals[0].text, "The person I am working with is stupid.")
        self.assertEqual(pipeline.analyzed, [finals[0].text])

    async def test_backlogged_audio_is_coalesced_into_one_latest_update(self) -> None:
        settings = Settings(
            _env_file=None,
            streaming_window_sec=0.35,
            streaming_update_sec=0.2,
            sentence_silence_sec=2.0,
            speech_rms_threshold=0.01,
        )
        pipeline = FakeStreamingPipeline()
        published: list[LiveEvent] = []

        async def publish(event: LiveEvent) -> None:
            published.append(event)

        worker = PipelineWorker(1, pipeline, publish, settings)
        chunk = np.full(int(16_000 * 0.1), 0.2, dtype=np.float32).tobytes()

        await worker._process_batch([chunk] * 20)

        self.assertEqual(pipeline.revision, 1)
        self.assertEqual(len(worker._buffer), len(chunk) * 20)
        self.assertEqual(
            [event.type for event in published], ["transcript.partial"]
        )

    async def test_final_update_failure_uses_last_good_partial_and_resets(self) -> None:
        settings = Settings(
            _env_file=None,
            streaming_window_sec=0.35,
            streaming_update_sec=0.2,
            sentence_silence_sec=0.8,
            speech_rms_threshold=0.01,
        )
        pipeline = FakeStreamingPipeline()
        published: list[LiveEvent] = []

        async def publish(event: LiveEvent) -> None:
            published.append(event)

        worker = PipelineWorker(1, pipeline, publish, settings)
        voice = np.full(int(16_000 * 0.4), 0.2, dtype=np.float32).tobytes()
        silence = np.zeros(int(16_000 * 0.8), dtype=np.float32).tobytes()
        await worker._handle_chunk(voice)
        pipeline.fail_next = True

        with self.assertLogs(
            "backend.app.pipelines.worker", level="ERROR"
        ):
            await worker._handle_chunk(silence)

        finals = [event for event in published if event.type == "transcript.final"]
        self.assertEqual([event.text for event in finals], [
            "The person I am working"
        ])
        self.assertEqual(worker._buffer, bytearray())


if __name__ == "__main__":
    unittest.main()
