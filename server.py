"""YuE2 HTTP server: async task API for the GPUStack facade.

Run:
    python3 -m uvicorn server:app --host 0.0.0.0 --port 8000

Env:
    YUE2_MODEL_DIR          default /weights   pipeline root holding pipeline.json,
                                               YuE2-3B/ and YuE2-Vae/, plus
                                               SheetSage2/ and MERT-v2-FullSong/
                                               for covers (GPUStack injects the
                                               model path here)
    YUE2_VAE_DIR            unset              explicit decoder dir; only needed when
                                               YUE2_MODEL_DIR is a bare YuE2-3B dir
    YUE2_SHEETSAGE2_DIR     <model>/SheetSage2 cover transcriber; cover is disabled
    YUE2_MERT_DIR           <model>/MERT-v2-FullSong   (400) when either is absent
    YUE2_MAX_QUEUE          default 8          per-instance FIFO; full -> 503
    YUE2_MEMORY_BUDGET_GIB  default 24         upstream's per-process VRAM cap (it
                                               reserves 2 GiB of it); measured peak
                                               is ~9 GiB on A100 regardless of length
    YUE2_MP3_BITRATE        default 320k       CBR bitrate for .mp3 outputs
    YUE2_WARMUP             default 1          run one tiny song before /ready

One door: the facade's music kind (POST /v1/tasks/music/), serving both t2m and
cover. YuE2 has no streaming, so unlike Breeze there is no sync endpoint to
keep. Beside the audio at save_result_path the worker writes two sidecars into
the same directory, which the facade's janitor reaps together with the audio
(it rmtree's whole day directories):
    <stem>.abc   the score the song was rendered from — planned by YuE2,
                 transcribed from a cover's source, or the one supplied (absent
                 for cot=off). Edit it and submit it back as ``abc`` to re-render.
    <stem>.json  seed, mode, token counts, timings: everything needed to
                 reproduce or explain the take
"""
import importlib
import importlib.util
import json
import logging
import os
import subprocess
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import soundfile as sf
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from harness.routes import create_tasks_router
from harness.schemas import MusicTaskRequest, lyric_units
from harness.task_manager import TaskManager
from harness.worker import TaskWorker
from yue2.protocol import SongRequest

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("yue2.server")

MODEL_DIR = os.environ.get("YUE2_MODEL_DIR", "/weights")
VAE_DIR = os.environ.get("YUE2_VAE_DIR") or None
MAX_QUEUE = int(os.environ.get("YUE2_MAX_QUEUE", "8"))
MEMORY_BUDGET_GIB = float(os.environ.get("YUE2_MEMORY_BUDGET_GIB", "24"))
MP3_BITRATE = os.environ.get("YUE2_MP3_BITRATE", "320k")
WARMUP = os.environ.get("YUE2_WARMUP", "1") not in {"0", "false", "False", ""}
SHEETSAGE2_DIR = os.environ.get("YUE2_SHEETSAGE2_DIR") or os.path.join(MODEL_DIR, "SheetSage2")
MERT_DIR = os.environ.get("YUE2_MERT_DIR") or os.path.join(MODEL_DIR, "MERT-v2-FullSong")

SAMPLE_RATE = 48000


@asynccontextmanager
async def _lifespan(_app):
    # Loading blocks the event loop on purpose: nothing may be served before the
    # weights are in and the warmup song has gone through.
    _load()
    yield
    _worker.stop()


app = FastAPI(title="YuE2 music task server", lifespan=_lifespan)

_pipe = None
_transcriber = None  # SheetSage2; stays None when the cover weights are absent
_task_manager = TaskManager(max_queue_size=MAX_QUEUE)


class TruncatedError(RuntimeError):
    """An AR stage hit its token cap: the song (or its score) is cut short.

    Upstream treats this as success and only sets a ``truncated`` flag, so the
    facade would publish a song that stops mid-phrase. Fail it instead; the
    submit-time lyric budget keeps this rare.
    """

    error_type = "truncated"


class TranscriptionError(RuntimeError):
    """SheetSage2 could not turn the cover's source into a usable score."""

    error_type = "transcription"


# ----------------------------------------------------------------- progress
# Global progress, folded here rather than by the facade: the facade's music
# table (prepare 5 / encode 10 / denoise 70 / decode 10 / save 5) does not match
# where YuE2's time goes, and its contract lets an engine-reported global
# ``progress`` win outright. Spans follow the measured A100 split across four
# 1.5-4 minute songs: score ~30 %, semantic ~50 %, flow matching ~12 %, VAE
# decode ~6 %, MP3 encode + NFS write the rest. Phase names stay in the facade's
# vocabulary (prepare / denoise / decode / save) as labels only.
# A cover spends its first ~20 % transcribing (A100: 5.7 s of a 45 s song's
# 33 s, 28 s of a 4-minute song's 115 s) and has no planning stage.
_TRANSCRIBE_END = 20.0
_PLAN_SPAN = (0.0, 30.0)
_SEMANTIC_SPAN = (30.0, 80.0)
_NAR_START = 80.0
_DECODE_START = 92.0
_SAVE_START = 97.0

# Token-count expectations, used only to pace the bar. AR stages decide their
# own length, so the true count is unknowable until they stop. Measured on A100
# over five songs (lyric length in harness.schemas.lyric_units): long lyrics
# settle at ~4.2 score tokens and ~11 semantic tokens per unit, while short ones
# get a longer arrangement than their length suggests — hence the floors. Once
# the score exists it is the better predictor: semantic runs 2.4-2.9x the score
# (a 45 s song: 405 -> 1119). The facade clamps an engine reading
# at 99 % and forces it monotonic, so an estimate that is off never moves the
# bar backwards; it only makes it pause or hurry.
_ABC_TOKENS_PER_UNIT = 4.3
_MIN_EXPECTED_ABC = 400
_SEMANTIC_TOKENS_PER_ABC_TOKEN = 2.7
_SEMANTIC_TOKENS_PER_UNIT_NO_PLAN = 11.2
_MIN_EXPECTED_SEMANTIC = 1000


class _Progress:
    """Maps AR token callbacks onto the task's global progress."""

    def __init__(self, task_id: str, units: float, planned: bool, start: float = 0.0):
        self.task_id = task_id
        self.planned = planned
        # Where the semantic stage starts when there is no planning stage in
        # front of it (a supplied score starts at 0, a cover after transcribing).
        self.start = start
        self.counts = {"abc": 0, "semantic": 0}
        self.expected = {
            "abc": max(_MIN_EXPECTED_ABC, units * _ABC_TOKENS_PER_UNIT),
            "semantic": max(_MIN_EXPECTED_SEMANTIC, units * _SEMANTIC_TOKENS_PER_UNIT_NO_PLAN),
        }

    def plan_finished(self, abc_tokens: int) -> None:
        if abc_tokens:
            self.expected["semantic"] = max(
                _MIN_EXPECTED_SEMANTIC, abc_tokens * _SEMANTIC_TOKENS_PER_ABC_TOKEN
            )

    def on_token(self, phase: str, _token: int) -> None:
        if phase not in self.counts:
            return
        self.counts[phase] += 1
        # ~105 tokens/s: reporting every token would take the manager lock a
        # hundred times a second for a bar the facade polls every few seconds.
        if self.counts[phase] % 16:
            return
        if phase == "abc":
            lo, hi = _PLAN_SPAN
        else:
            lo, hi = _SEMANTIC_SPAN if self.planned else (self.start, _SEMANTIC_SPAN[1])
        fraction = min(1.0, self.counts[phase] / self.expected[phase])
        _task_manager.set_progress(self.task_id, "denoise", lo + (hi - lo) * fraction)


# ------------------------------------------------------------------- output
def _part_path(path: str) -> str:
    """Same-directory temp path that keeps the extension last.

    soundfile and ffmpeg both pick the container from the suffix, so a bare
    "<final>.mp3.part" would be refused ("unable to get format from file
    extension" — the trap index-tts and Breeze both hit)."""
    root, ext = os.path.splitext(path)
    return f"{root}.part{ext}"


def _write_audio(audio: np.ndarray, path: str) -> None:
    ext = os.path.splitext(path)[1].lower()
    if ext in {".wav", ".flac"}:
        sf.write(path, audio, SAMPLE_RATE, subtype="PCM_24")
        return
    # MP3 through ffmpeg/LAME with an explicit CBR bitrate. The master is
    # float32 48 kHz stereo; piping raw PCM avoids a lossless temp file.
    pcm = np.ascontiguousarray(audio, dtype=np.float32)
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "f32le", "-ar", str(SAMPLE_RATE), "-ac", str(pcm.shape[1]), "-i", "pipe:0",
        "-codec:a", "libmp3lame", "-b:a", MP3_BITRATE, path,
    ]
    result = subprocess.run(command, input=pcm.tobytes(), capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg mp3 encode failed: {result.stderr.decode(errors='replace').strip()}")


def _atomic_text(path: str, text: str) -> None:
    tmp = _part_path(path)
    try:
        Path(tmp).write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def _publish(audio: np.ndarray, save_result_path: str, score: Optional[str], record: dict) -> None:
    """Write sidecars first, audio last, each via temp + os.replace.

    The audio's appearance at its final path is the completion signal; ordering
    it last means a reader that sees the song can always find its score."""
    os.makedirs(os.path.dirname(save_result_path) or ".", exist_ok=True)
    root = os.path.splitext(save_result_path)[0]
    if score is not None:
        _atomic_text(root + ".abc", score)
    _atomic_text(root + ".json", json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    tmp = _part_path(save_result_path)
    try:
        _write_audio(audio, tmp)
        os.replace(tmp, save_result_path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


# ------------------------------------------------------------------- cover
def _load_abc_tools():
    """Upstream's ABC checker, which ships with the agent skill, not the package."""
    path = Path(__file__).resolve().parent / "skills" / "yue2-music" / "scripts" / "abc_tools.py"
    spec = importlib.util.spec_from_file_location("yue2_abc_tools", path)
    module = importlib.util.module_from_spec(spec)
    # Registered before exec: its dataclasses resolve string annotations through
    # sys.modules[cls.__module__] and crash on a module that is not there.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_transcriber():
    """SheetSage2 on the GPU, or None when the cover weights are not installed.

    Imported as a regular package from the snapshot directory rather than
    through AutoModel(trust_remote_code=True): transformers 4.57 copies only a
    model file's direct relative imports into its dynamic-module cache, and
    SheetSage2 has second-level ones, so the remote-code route dies on a missing
    chord_spelling_sheetsage2.py. Its pins (torch 2.8, transformers 4.45, numpy
    1.24) are the authors' environment, not requirements: it runs unchanged on
    this image's torch 2.11 / transformers 4.57.6 / numpy 2.5 (measured)."""
    if not (os.path.isdir(SHEETSAGE2_DIR) and os.path.isdir(MERT_DIR)):
        logger.info(f"Cover disabled: {SHEETSAGE2_DIR} or {MERT_DIR} not found")
        return None
    parent, package = os.path.split(os.path.abspath(SHEETSAGE2_DIR))
    if parent not in sys.path:
        sys.path.insert(0, parent)
    module = importlib.import_module(f"{package}.modeling_sheetsage2")
    # base_model_path keeps the MERT-v2 parent offline; the loader still checks
    # it against the sha256 pinned in SheetSage2's config.
    return module.SheetSage2Model.from_pretrained(
        SHEETSAGE2_DIR, local_files_only=True, base_model_path=MERT_DIR,
    ).eval().to("cuda")


def _transcribe(path: str, keep_chords: bool) -> tuple:
    """Source recording -> ABC score, validated like upstream's transcribe helper."""
    try:
        result = _transcriber.transcribe(path, melody_only=not keep_chords)
    except RuntimeError as e:
        # melody_only raises when the melody-only score cannot be built.
        raise TranscriptionError(f"transcription failed: {e}") from e
    abc = result.get("abc")
    if not abc or result.get("abc_error"):
        raise TranscriptionError(f"transcription produced no usable score: {result.get('abc_error')}")
    try:
        score = _abc_tools.parse_abc(abc)
    except Exception as e:  # noqa: BLE001 - the checker raises plain errors on bad notation
        raise TranscriptionError(f"transcribed score failed the ABC check: {e}") from e
    if not keep_chords and any(voice.chords for voice in score.voices.values()):
        raise TranscriptionError("melody transcription unexpectedly contains chord symbols")
    return abc, result.get("warnings", [])


_abc_tools = _load_abc_tools()


# --------------------------------------------------------------- generation
def _generate(req: MusicTaskRequest, cancelled: Callable[[], bool]) -> None:
    """Blocking generation for one task; writes req.save_result_path.

    Raises InterruptedError when cancelled (from inside the pipeline, within
    one token / flow step), TruncatedError when a stage hits its cap."""
    if _pipe is None:
        raise RuntimeError("model not loaded")
    task_id = req.task_id or ""
    started = time.perf_counter()

    _task_manager.set_progress(task_id, "prepare")
    cot = req.cot_mode()
    abc = req.abc_text()
    cover = None
    semantic_start = 0.0
    if req.is_cover():
        if _transcriber is None:
            raise TranscriptionError("cover requested but SheetSage2 is not loaded")
        _task_manager.set_progress(task_id, "encode", 0.0)
        transcribe_started = time.perf_counter()
        # cot=full keeps the source's harmony; melody (the default) drops the
        # chord symbols so the new style chooses its own.
        abc, warnings = _transcribe(req.reference_audio_path, keep_chords=cot == "full")
        cover = {"source_audio": req.reference_audio_path, "transcription_warnings": warnings,
                 "transcribe_seconds": round(time.perf_counter() - transcribe_started, 3)}
        if cancelled():
            raise InterruptedError("Cancelled after transcription")
        _task_manager.set_progress(task_id, "encode", _TRANSCRIBE_END)
        semantic_start = _TRANSCRIBE_END

    request = SongRequest(style=req.style_text(), lyrics=req.lyrics_text(), cot=cot,
                          seed=req.seed, abc=abc, cfg_scale=req.cfg_scale, id="task")
    planned = request.cot != "off" and request.abc is None
    progress = _Progress(task_id, lyric_units(request.lyrics), planned, start=semantic_start)

    plan = _pipe.plan(request=request, cancelled=cancelled, on_token=progress.on_token)
    if plan.truncated:
        raise TruncatedError("score planning hit its 4096-token cap; shorten the lyrics")
    # A planned, supplied or transcribed score is the best predictor of the
    # semantic length alike.
    progress.plan_finished(len(plan.abc_ids))

    semantic = _pipe.generate_semantic(plan, cancelled=cancelled, on_token=progress.on_token)
    if semantic.truncated:
        raise TruncatedError(
            "song hit the 9000-token (~6 min) ceiling and would end mid-phrase; shorten the lyrics"
        )

    _task_manager.set_progress(task_id, "denoise", _NAR_START)
    nar_started = time.perf_counter()
    latents = _pipe.synthesize(semantic, cancelled=cancelled)
    nar_seconds = time.perf_counter() - nar_started

    if cancelled():
        raise InterruptedError("Cancelled before VAE")
    _task_manager.set_progress(task_id, "decode", _DECODE_START)
    vae_started = time.perf_counter()
    audio = _pipe.decode(latents)
    vae_seconds = time.perf_counter() - vae_started

    _task_manager.set_progress(task_id, "save", _SAVE_START)
    audio_seconds = len(audio) / SAMPLE_RATE
    record = {
        "engine": "yue2",
        "seed": request.seed,
        "cot": request.cot,
        "cfg_scale": request.guidance,
        "score_source": ("transcribed" if cover else "planned" if planned
                         else "provided" if request.abc is not None else "none"),
        "sample_rate": SAMPLE_RATE,
        "channels": int(audio.shape[1]),
        "audio_seconds": round(audio_seconds, 3),
        "abc_tokens": len(plan.abc_ids),
        "semantic_tokens": len(semantic.tokens),
        "timing": {
            "plan": plan.timing,
            "semantic": semantic.timing,
            "nar_seconds": round(nar_seconds, 3),
            "vae_seconds": round(vae_seconds, 3),
            "generate_seconds": round(time.perf_counter() - started, 3),
        },
        "weights": {name: {file: entry["sha256"] for file, entry in identity["files"].items()}
                    for name, identity in _pipe.weights.items()},
    }
    if cover is not None:
        record["cover"] = cover
    _publish(audio, req.save_result_path, plan.abc, record)
    logger.info(
        f"Task {task_id}: {audio_seconds:.1f}s audio in {record['timing']['generate_seconds']:.1f}s "
        f"(cot={request.cot}, abc={len(plan.abc_ids)}, semantic={len(semantic.tokens)})"
    )


def _warmup(pipe) -> None:
    """Run one tiny song end to end before /ready.

    YuE2 captures its CUDA graphs per request, so there is no long capture to
    pay up front like Breeze's; this exists to initialise CUDA/cuBLAS, load the
    VAE from disk and — mostly — to prove the whole chain works in this image
    before the scheduler routes a user at it."""
    request = SongRequest(style="pop, piano", lyrics="[Verse]\nla la la", cot="off", seed=0, id="warmup")
    plan = pipe.plan(request=request)
    semantic = pipe.generate_semantic(plan, sampling={"max_tokens": 64, "min_tokens": 32})
    pipe.decode(pipe.synthesize(semantic))


_worker = TaskWorker(_task_manager, _generate)
app.include_router(create_tasks_router(_task_manager, cover_enabled=lambda: _transcriber is not None))


def _load() -> None:
    global _pipe, _transcriber
    from yue2 import YuE2Pipeline

    started = time.time()
    kwargs = {"vae": VAE_DIR} if VAE_DIR else {}
    pipe = YuE2Pipeline.from_pretrained(
        MODEL_DIR, local_files_only=True, progress=False, device="cuda",
        memory_budget_gib=MEMORY_BUDGET_GIB, **kwargs,
    )
    logger.info(f"YuE2 weights loaded in {time.time() - started:.1f}s from {MODEL_DIR}")
    # After YuE2, so upstream's per-process VRAM cap already covers both models
    # (~9 + ~3.4 GiB peak on A100).
    transcriber_started = time.time()
    transcriber = _load_transcriber()
    if transcriber is not None:
        logger.info(f"SheetSage2 loaded in {time.time() - transcriber_started:.1f}s; cover enabled")
    if WARMUP:
        warm = time.time()
        _warmup(pipe)
        logger.info(f"YuE2 warmup took {time.time() - warm:.1f}s")
    _pipe = pipe
    _transcriber = transcriber
    _worker.start()
    logger.info(f"YuE2 ready in {time.time() - started:.1f}s")


@app.get("/ready")
def ready():
    if _pipe is None:
        return JSONResponse(status_code=503, content={"status": "loading"})
    return {"status": "ready", "sample_rate": SAMPLE_RATE, "cover": _transcriber is not None}


@app.get("/health")
def health():
    return {"status": "ok" if _pipe is not None else "loading"}
