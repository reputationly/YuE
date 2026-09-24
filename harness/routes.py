"""HTTP task routes for the YuE2 async task server.

Contract mirrors LightX2V / ACE-Step so the GPUStack facade can
dispatch/poll/cancel without special-casing YuE2:

    POST   /v1/tasks/music/          -> {task_id, task_status, save_result_path}   (503 when queue full)
    POST   /v1/tasks/audio/          -> alias, same as ACE-Step keeps for manual testing
    GET    /v1/tasks/queue/status    -> {is_processing, current_task, pending_count, active_count, queue_size, queue_available}
    GET    /v1/tasks/{id}/status     -> {task_id, status, error, error_type, save_result_path[, phase, progress]}   (404 if unknown)
    GET    /v1/tasks/{id}/result     -> streams the audio (COMPLETED only)
    DELETE /v1/tasks/{id}            -> {stop_status, reason}

The queue itself is YuE2-Turbo's JobStore (the worker's SQLite queue); these
routes translate its states into the facade's vocabulary. Note: /queue/status
is registered before /{task_id}/status so "queue" is not captured as a task_id.
"""
import json
import logging
import os
import secrets
import subprocess
import threading
from datetime import datetime
from typing import Callable, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from yue2.protocol import SongRequest
from yue2.service_store import QueueFull, TERMINAL

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

logger = logging.getLogger(__name__)

# Private request keys stored with each job. Turbo's generation_fields() passes
# only the pipeline's own fields to YuE2, so these never reach the model.
SAVE_KEY = "_save_result_path"
REF_KEY = "_reference_audio_path"

# Output container follows save_result_path's extension. The facade always asks
# for .mp3 on the music kind; wav/flac are there for direct callers that want
# the lossless master.
MEDIA_TYPES = {".mp3": "audio/mpeg", ".wav": "audio/wav", ".flac": "audio/flac"}

# Turbo's job states -> the facade's _ENGINE_STATE_MAP vocabulary (verbatim;
# cancelled is double-L). "truncated" never reaches a caller: the worker fails
# a truncated song before it is published.
FACADE_STATUS = {
    "queued": "pending",
    "running": "processing",
    "succeeded": "completed",
    "truncated": "failed",
    "failed": "failed",
    "cancelled": "cancelled",
}


# ------------------------------------------------------------------ progress
# Global progress, folded here rather than by the facade: the facade's music
# table (prepare 5 / encode 10 / denoise 70 / decode 10 / save 5) does not match
# where YuE2's time goes, and its contract lets an engine-reported global
# ``progress`` win outright. Spans follow the measured A100 split: score ~30 %,
# semantic ~50 %, flow matching ~12 %, VAE decode ~6 %, MP3 encode + NFS write
# the rest. A cover spends its first ~20 % transcribing and has no planning
# stage. Phase names stay in the facade's vocabulary as labels only.
_TRANSCRIBE_END = 20.0
_PLAN_SPAN = (0.0, 30.0)
_SEMANTIC_END = 80.0
_STAGE_FLOOR = {"synthesis": 80.0, "decode": 92.0, "saving": 97.0}
_STAGE_PHASE = {
    "claimed_waiting": "prepare", "transcribing": "encode", "planning": "denoise",
    "semantic": "denoise", "synthesis": "denoise", "decode": "decode", "saving": "save",
}

# Token-count expectations, used only to pace the bar. AR stages decide their
# own length, so the true count is unknowable until they stop. Measured on A100
# over five songs (lyric length in harness.schemas.lyric_units): long lyrics
# settle at ~4.2 score tokens and ~11 semantic tokens per unit, while short ones
# get a longer arrangement than their length suggests — hence the floors. Once
# the score exists it is the better predictor: semantic runs 2.4-2.9x the score.
# The facade clamps an engine reading at 99 % and forces it monotonic, so an
# estimate that is off never moves the bar backwards; it only pauses or hurries.
_ABC_TOKENS_PER_UNIT = 4.3
_MIN_EXPECTED_ABC = 400
_SEMANTIC_TOKENS_PER_ABC_TOKEN = 2.7
_SEMANTIC_TOKENS_PER_UNIT_NO_PLAN = 11.2
_MIN_EXPECTED_SEMANTIC = 1000


class ProgressBook:
    """Per-task facts the progress fold needs and the job store does not keep:
    lyric length, whether YuE2 plans the score, whether it is a cover, and the
    high-water mark that keeps the reported value from ever moving backwards.
    Lives as long as the process, like the store it shadows."""

    def __init__(self):
        self._lock = threading.Lock()
        self._tasks = {}

    def register(self, task_id: str, *, units: float, planned: bool, cover: bool, save_result_path: str):
        with self._lock:
            self._tasks[task_id] = {"units": units, "planned": planned, "cover": cover,
                                    "save_result_path": save_result_path, "high": None}

    def get(self, task_id: str) -> Optional[dict]:
        with self._lock:
            return self._tasks.get(task_id)

    def fold(self, task_id: str, job: dict):
        """(phase, progress) for a running job, progress None when there is
        nothing to say yet. Never lower than a value reported before."""
        stage = job.get("stage")
        phase = _STAGE_PHASE.get(stage)
        with self._lock:
            meta = self._tasks.get(task_id)
            if meta is None or phase is None:
                return phase, None
            value = _raw_progress(stage, job.get("tokens") or {}, meta)
            if value is not None:
                meta["high"] = value if meta["high"] is None else max(meta["high"], value)
            high = meta["high"]
        return phase, (None if high is None else round(min(99.0, high), 1))


def _raw_progress(stage: str, tokens: dict, meta: dict) -> Optional[float]:
    if stage in _STAGE_FLOOR:
        return _STAGE_FLOOR[stage]
    units = meta["units"]
    if stage == "planning":
        expected = max(_MIN_EXPECTED_ABC, units * _ABC_TOKENS_PER_UNIT)
        lo, hi = _PLAN_SPAN
        return lo + (hi - lo) * min(1.0, tokens.get("abc", 0) / expected)
    if stage == "semantic":
        abc_tokens = tokens.get("abc", 0)
        if meta["planned"] and abc_tokens:
            expected = max(_MIN_EXPECTED_SEMANTIC, abc_tokens * _SEMANTIC_TOKENS_PER_ABC_TOKEN)
        else:
            expected = max(_MIN_EXPECTED_SEMANTIC, units * _SEMANTIC_TOKENS_PER_UNIT_NO_PLAN)
        start = _PLAN_SPAN[1] if meta["planned"] else (_TRANSCRIBE_END if meta["cover"] else 0.0)
        return start + (_SEMANTIC_END - start) * min(1.0, tokens.get("semantic", 0) / expected)
    # claimed_waiting / transcribing: nothing measurable yet (the facade falls
    # back to its elapsed-time estimate).
    return None


# ---------------------------------------------------------------- validation
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


def _job_request(message: MusicTaskRequest) -> dict:
    request = {"style": message.style_text(), "lyrics": message.lyrics_text(),
               "cot": message.cot_mode(), "seed": message.seed, SAVE_KEY: message.save_result_path}
    if message.abc_text() is not None:
        request["abc"] = message.abc_text()
    if message.cfg_scale is not None:
        request["cfg_scale"] = message.cfg_scale
    if message.is_cover():
        request[REF_KEY] = message.reference_audio_path
    return request


def _timestamp(value):
    return datetime.fromtimestamp(value) if value else None


# -------------------------------------------------------------------- routes
def create_tasks_router(get_worker: Callable[[], object], book: ProgressBook) -> APIRouter:
    """``get_worker`` returns the running HarnessWorker (None before startup):
    the router is built at import time, the worker only in the app lifespan."""
    router = APIRouter(prefix="/v1/tasks", tags=["tasks"])

    def worker_or_503():
        worker = get_worker()
        if worker is None or not worker.ready:
            raise HTTPException(status_code=503, detail="engine is still loading")
        return worker

    # The facade posts to /v1/tasks/{engine_kind}/; t2m and cover both resolve
    # to "music".
    @router.post("/music/", response_model=TaskResponse)
    @router.post("/audio/", response_model=TaskResponse)
    async def create_music_task(message: MusicTaskRequest):
        worker = worker_or_503()
        validate_request(message, worker.cover_transcriber is not None)
        # Pin the seed at submit time so a retried or re-read task is
        # reproducible and the sidecar records what was actually used. Upstream
        # defaults every request to seed 831001, which would make the same
        # prompt produce the same song for every caller.
        if message.seed is None:
            message.seed = secrets.randbelow(2**31)
        request = _job_request(message)
        try:
            job, _ = worker.store.submit(request, worker.settings.max_pending)
        except QueueFull as e:  # queue full -> backpressure (retryable)
            raise HTTPException(status_code=503, detail=f"Task queue is full (max {worker.settings.max_pending} tasks)") from e
        book.register(job["id"], units=lyric_units(request["lyrics"]),
                      planned=request["cot"] != "off" and "abc" not in request and not message.is_cover(),
                      cover=message.is_cover(), save_result_path=message.save_result_path)
        worker.wake.set()
        return TaskResponse(task_id=job["id"], task_status="pending",
                            save_result_path=message.save_result_path)

    @router.get("/queue/status")
    async def get_queue_status():
        worker = get_worker()
        if worker is None:
            raise HTTPException(status_code=503, detail="engine is still loading")
        snapshot = worker.store.load_snapshot()
        running = [item["id"] for item in snapshot["items"] if item["stage"] != "queued"]
        active = snapshot["active_requests"] + snapshot["queue_depth"]
        size = worker.settings.max_pending
        return {
            "is_processing": snapshot["active_requests"] > 0,
            "current_task": running[0] if running else None,
            "pending_count": snapshot["queue_depth"],
            "active_count": active,
            "queue_size": size,
            "queue_available": size - active,
        }

    def facade_status(task_id: str) -> Optional[dict]:
        worker = get_worker()
        job = worker.store.get(task_id) if worker is not None else None
        if job is None:
            return None
        meta = book.get(task_id) or {}
        status = FACADE_STATUS.get(job["status"], "processing")
        # A DELETE is immediate from the caller's side (the old in-memory queue
        # flipped to cancelled at once); the worker notices within one token or
        # flow step, or after the current transcription.
        if job.get("cancel_requested") and job["status"] not in TERMINAL:
            status = "cancelled"
        error = job.get("error") or {}
        payload = {
            "task_id": task_id,
            "status": status,
            "start_time": _timestamp(job.get("started_at")),
            "end_time": _timestamp(job.get("finished_at")),
            "error": error.get("message") or ("Task cancelled by user" if status == "cancelled" else None),
            "error_type": error.get("code") or "",
            "save_result_path": (job.get("result") or {}).get("save_result_path") or meta.get("save_result_path"),
        }
        if job["status"] == "running" and status == "processing":
            phase, progress = book.fold(task_id, job)
            if phase is not None:
                payload["phase"] = phase
            if progress is not None:
                payload["progress"] = progress
        return payload

    @router.get("/{task_id}/status")
    async def get_task_status(task_id: str):
        status = facade_status(task_id)
        if not status:
            # 404 is how the facade sweeper detects a lost engine task (instance
            # restarted) and re-dispatches it. Any other code strands the task.
            raise HTTPException(status_code=404, detail="Task not found")
        return status

    @router.get("/{task_id}/result")
    async def get_task_result(task_id: str):
        status = facade_status(task_id)
        if not status:
            raise HTTPException(status_code=404, detail="Task not found")
        if status["status"] != "completed":
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
        worker = get_worker()
        try:
            job = worker.store.get(task_id) if worker is not None else None
            if job is None or job["status"] in TERMINAL or job.get("cancel_requested"):
                return StopTaskResponse(stop_status="do_nothing", reason="Task not found or already completed.")
            worker.cancel(task_id)
            logger.info(f"Task {task_id} cancelled.")
            return StopTaskResponse(stop_status="success", reason="Task stopped successfully.")
        except Exception as e:  # noqa: BLE001
            logger.error(f"Error cancelling task {task_id}: {e}")
            return StopTaskResponse(stop_status="error", reason=str(e))

    return router
