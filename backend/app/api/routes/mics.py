"""Microphone pipeline control endpoints."""

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from backend.app.pipelines.manager import PipelineManager

router = APIRouter(prefix="/api/mics", tags=["microphones"])


class PipelineStatusResponse(BaseModel):
    """Pipeline status response payload."""

    microphone_id: int
    status: str


def get_manager(request: Request) -> PipelineManager:
    """Retrieve the application-owned pipeline manager."""
    return request.app.state.pipeline_manager


def validate_microphone(microphone_id: int) -> None:
    """Reject microphone identifiers outside the two configured inputs."""
    if microphone_id not in (1, 2):
        raise HTTPException(status_code=404, detail="Microphone not found")


@router.get("/{microphone_id}", response_model=PipelineStatusResponse)
async def get_status(microphone_id: int, request: Request) -> PipelineStatusResponse:
    """Return a microphone pipeline's current state."""
    validate_microphone(microphone_id)
    return PipelineStatusResponse(
        microphone_id=microphone_id, status=get_manager(request).status(microphone_id)
    )


@router.post("/{microphone_id}/start", response_model=PipelineStatusResponse)
async def start_pipeline(
    microphone_id: int, request: Request
) -> PipelineStatusResponse:
    """Start processing for one microphone only."""
    validate_microphone(microphone_id)
    return PipelineStatusResponse(
        microphone_id=microphone_id, status=get_manager(request).start(microphone_id)
    )


@router.post("/{microphone_id}/stop", response_model=PipelineStatusResponse)
async def stop_pipeline(microphone_id: int, request: Request) -> PipelineStatusResponse:
    """Stop processing for one microphone only."""
    validate_microphone(microphone_id)
    return PipelineStatusResponse(
        microphone_id=microphone_id,
        status=await get_manager(request).stop(microphone_id),
    )
