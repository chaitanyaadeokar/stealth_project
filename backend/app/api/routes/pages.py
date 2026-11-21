"""Server-rendered podcast pages."""

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from backend.app.pipelines.manager import PipelineManager

router = APIRouter()
templates = Jinja2Templates(directory="backend/app/templates")


@router.get("/api/live")
async def live_snapshot(request: Request) -> list[dict]:
    """Read the shared conversation even if a browser cannot open WebSockets."""
    manager: PipelineManager = request.app.state.pipeline_manager
    return [event.model_dump(mode="json") for event in manager.live_snapshot()]


@router.get("/", response_class=HTMLResponse)
async def microphone_selection(request: Request) -> HTMLResponse:
    """Render the microphone selection page."""
    return templates.TemplateResponse(request, "index.html")


@router.get("/mic/{microphone_id}", response_class=HTMLResponse)
async def microphone_page(request: Request, microphone_id: int) -> HTMLResponse:
    """Render the selected microphone's independent pipeline state."""
    if microphone_id not in (1, 2):
        raise HTTPException(status_code=404, detail="Microphone not found")
    manager: PipelineManager = request.app.state.pipeline_manager
    return templates.TemplateResponse(
        request,
        "mic.html",
        {"microphone_id": microphone_id, "status": manager.status(microphone_id)},
    )


@router.get("/live", response_class=HTMLResponse)
async def live_page(request: Request) -> HTMLResponse:
    """Render the combined two-microphone transcript view."""
    return templates.TemplateResponse(request, "live.html")
