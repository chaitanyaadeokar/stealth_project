"""Live microphone WebSocket endpoint; audio in and events out share one socket."""

import asyncio
import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from backend.app.pipelines.manager import PipelineManager

router = APIRouter()
VALID_STREAM_IDS = (0, 1, 2)
MICROPHONE_IDS = (1, 2)


@router.websocket("/ws/mic/{mic_id}")
async def microphone_socket(websocket: WebSocket, mic_id: int) -> None:
    """Receive PCM float32/16kHz and send structured live events."""
    if mic_id not in VALID_STREAM_IDS:
        await websocket.close(code=1008)
        return
    await websocket.accept()
    manager: PipelineManager = websocket.app.state.pipeline_manager
    event_queue = manager.subscribe(mic_id)
    send_lock = asyncio.Lock()

    async def send_events() -> None:
        """Forward worker events to this connected browser."""
        while True:
            event = await event_queue.get()
            async with send_lock:
                await websocket.send_json(event.model_dump(mode="json"))

    sender_task = asyncio.create_task(send_events())
    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                raise WebSocketDisconnect(message.get("code", 1000))
            data = message.get("bytes")
            if data is not None and mic_id in MICROPHONE_IDS:
                await manager.submit_audio(mic_id, data)
                continue
            text = message.get("text")
            if text is None or mic_id not in MICROPHONE_IDS:
                continue
            try:
                command = json.loads(text)
            except json.JSONDecodeError:
                continue
            if command.get("action") == "flush":
                await manager.flush_audio(mic_id)
                async with send_lock:
                    await websocket.send_json(
                        {
                            "type": "audio.flushed",
                            "request_id": command.get("request_id"),
                        }
                    )
    except WebSocketDisconnect:
        # The browser closed the socket; cleanup happens below.
        pass
    finally:
        sender_task.cancel()
        manager.unsubscribe(mic_id, event_queue)
