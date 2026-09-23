"""HTTP task routes for the YuE2 async task server.

Contract mirrors LightX2V / ACE-Step so the GPUStack facade can
dispatch/poll/cancel without special-casing YuE2:

    POST   /v1/tasks/music/          -> {task_id, task_status, save_result_path}   (503 when queue full)
    POST   /v1/tasks/audio/          -> alias, same as ACE-Step keeps for manual testing
    GET    /v1/tasks/queue/status    -> {is_processing, current_task, pending_count, active_count, queue_size, queue_available}
    GET    /v1/tasks/{id}/status     -> {task_id, status, error, error_type, save_result_path[, phase, progress]}   (404 if unknown)
    GET    /v1/tasks/{id}/result     -> streams the audio (COMPLETED only)
    DELETE /v1/tasks/{id}            -> {stop_status, reason}

Note: /queue/status is registered before /{task_id}/status so "queue" is not
captured as a task_id.
"""
import json
import logging
import os
import secrets
import subprocess
from typing import Callable, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from yue2.protocol import SongRequest

from .schemas import (
    MAX_ABC_CHARS,
    MAX_COVER_SOURCE_SECONDS,
    MAX_LYRIC_UNITS,
    MAX_STYLE_CHARS,
    MusicTaskRequest,
    StopTaskResponse,
    TaskResponse,
    lyric_units,
)
from .task_manager import QueueFullError, TaskManager, TaskStatus

logger = logging.getLogger(__name__)

# Output container follows save_result_path's extension. The facade always asks
# for .mp3 on the music kind; wav/flac are there for direct callers that want
# the lossless master.
MEDIA_TYPES = {".mp3": "audio/mpeg", ".wav": "audio/wav", ".flac": "audio/flac"}


def probe_duration(path: str) -> Optional[float]:
    """Duration in seconds via ffprobe, or None when the file cannot be read."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", path],
            capture_output=True, text=True, timeout=20,
        )
        return float(json.loads(result.stdout)["format"]["duration"]) if result.returncode == 0 else None
    except (OSError, ValueError, KeyError, subprocess.TimeoutExpired):
        return None


def validate_cover(message: MusicTaskRequest, cover_enabled: bool) -> None:
    if not cover_enabled:
        raise HTTPException(
            status_code=400,
            detail="cover is not available on this instance: SheetSage2 / MERT-v2-FullSong are not installed under the model directory",
        )
    if message.abc_text() is not None:
        raise HTTPException(
            status_code=400,
            detail="a cover transcribes its score from reference_audio; do not also send abc",
        )
    if message.cot_mode() == "off":
        raise HTTPException(status_code=400, detail="a cover renders a transcribed score; cot must be melody or full")
    path = message.reference_audio_path
    if not os.path.isfile(path):
        raise HTTPException(status_code=400, detail=f"reference_audio_path does not exist: {path}")
    seconds = probe_duration(path)
    if seconds is None:
        raise HTTPException(status_code=400, detail="reference audio cannot be decoded")
    if seconds > MAX_COVER_SOURCE_SECONDS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"reference audio too long ({seconds:.0f} > {MAX_COVER_SOURCE_SECONDS} s); a cover is about "
                "as long as its source and YuE2 caps a song at about six minutes"
            ),
        )


def validate_request(message: MusicTaskRequest, cover_enabled: bool = False) -> None:
    """Reject at submit time everything that would otherwise fail on the GPU.

    Raises HTTPException(400). Runs YuE2's own SongRequest validation too, so a
    bad cot/abc/cfg combination is a 400 now instead of a failed task later.
    """
    if not message.save_result_path:
        raise HTTPException(status_code=400, detail="save_result_path is required")
    if message.src_audio_path:
        raise HTTPException(
            status_code=400,
            detail="repaint (src_audio) is not supported: YuE2 has no local inpainting; re-render the whole song",
        )
    ext = os.path.splitext(message.save_result_path)[1].lower()
    if ext not in MEDIA_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"save_result_path must end in one of {sorted(MEDIA_TYPES)}, got {ext or 'no extension'!r}",
        )

    style = message.style_text()
    if not style:
        raise HTTPException(status_code=400, detail="prompt (style description) is required")
    if len(style) > MAX_STYLE_CHARS:
        raise HTTPException(
            status_code=400,
            detail=f"prompt too long ({len(style)} > {MAX_STYLE_CHARS} chars)",
        )

    lyrics = message.lyrics_text()
    units = lyric_units(lyrics)
    if units > MAX_LYRIC_UNITS:
        # Not a taste call: past this the 9000-token semantic ceiling (~6 min of
        # audio) cuts the song off mid-phrase without any upstream error.
        raise HTTPException(
            status_code=400,
            detail=(
                f"lyrics too long ({units:.0f} > {MAX_LYRIC_UNITS} CJK-equivalent chars; "
                "Latin letters count ~0.4 each) — YuE2 caps a song at about six minutes"
            ),
        )

    if message.is_cover():
        validate_cover(message, cover_enabled)

    abc = message.abc_text()
    if abc is not None and len(abc) > MAX_ABC_CHARS:
        raise HTTPException(
            status_code=400, detail=f"abc too long ({len(abc)} > {MAX_ABC_CHARS} chars)"
        )

    try:
        SongRequest(style=style, lyrics=lyrics, cot=message.cot_mode(), seed=message.seed or 0,
                    abc=abc, cfg_scale=message.cfg_scale)
    except (TypeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))


def create_tasks_router(
    task_manager: TaskManager, cover_enabled: Callable[[], bool] = lambda: False
) -> APIRouter:
    router = APIRouter(prefix="/v1/tasks", tags=["tasks"])

    # The facade posts to /v1/tasks/{engine_kind}/; t2m and cover both resolve
    # to "music".
    @router.post("/music/", response_model=TaskResponse)
    @router.post("/audio/", response_model=TaskResponse)
    async def create_music_task(message: MusicTaskRequest):
        validate_request(message, cover_enabled())
        # Pin the seed at submit time so a retried or re-read task is
        # reproducible and the sidecar records what was actually used. Upstream
        # defaults every request to seed 831001, which would make the same
        # prompt produce the same song for every caller.
        if message.seed is None:
            message.seed = secrets.randbelow(2**31)
        try:
            task_id = task_manager.create_task(message)
        except QueueFullError as e:  # queue full -> backpressure (retryable)
            raise HTTPException(status_code=503, detail=str(e))
        except RuntimeError as e:  # e.g. duplicate task id -> client error, no retry
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:  # noqa: BLE001
            logger.error(f"Failed to create music task: {e}")
            raise HTTPException(status_code=500, detail=str(e))
        message.task_id = task_id
        return TaskResponse(
            task_id=task_id,
            task_status="pending",
            save_result_path=message.save_result_path,
        )

    @router.get("/queue/status")
    async def get_queue_status():
        active = task_manager.get_active_task_count()
        return {
            "is_processing": task_manager.is_processing(),
            "current_task": task_manager.current_task,
            "pending_count": task_manager.get_pending_task_count(),
            "active_count": active,
            "queue_size": task_manager.max_queue_size,
            "queue_available": task_manager.max_queue_size - active,
        }

    @router.get("/{task_id}/status")
    async def get_task_status(task_id: str):
        status = task_manager.get_task_status(task_id)
        if not status:
            # 404 is how the facade sweeper detects a lost engine task (instance
            # restarted) and re-dispatches it. Any other code strands the task.
            raise HTTPException(status_code=404, detail="Task not found")
        return status

    @router.get("/{task_id}/result")
    async def get_task_result(task_id: str):
        status = task_manager.get_task_status(task_id)
        if not status:
            raise HTTPException(status_code=404, detail="Task not found")
        if status.get("status") != TaskStatus.COMPLETED.value:
            raise HTTPException(status_code=404, detail="Task not completed")
        path = status.get("save_result_path")
        if not path or not os.path.isfile(path):
            raise HTTPException(status_code=404, detail="Task result file does not exist")

        def _stream(p: str, chunk: int = 1 << 20):
            with open(p, "rb") as f:
                while data := f.read(chunk):
                    yield data

        media_type = MEDIA_TYPES.get(os.path.splitext(path)[1].lower(), "application/octet-stream")
        headers = {"Content-Disposition": f'attachment; filename="{os.path.basename(path)}"'}
        return StreamingResponse(_stream(path), media_type=media_type, headers=headers)

    @router.delete("/{task_id}", response_model=StopTaskResponse)
    async def stop_task(task_id: str):
        try:
            if task_manager.cancel_task(task_id):
                logger.info(f"Task {task_id} cancelled.")
                return StopTaskResponse(stop_status="success", reason="Task stopped successfully.")
            return StopTaskResponse(stop_status="do_nothing", reason="Task not found or already completed.")
        except Exception as e:  # noqa: BLE001
            logger.error(f"Error cancelling task {task_id}: {e}")
            return StopTaskResponse(stop_status="error", reason=str(e))

    return router
