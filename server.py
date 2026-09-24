"""YuE2 HTTP server: GPUStack async task API over YuE2-Turbo's scheduler.

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
    YUE2_BACKEND            default vllm       vllm = accelerated; torch = upstream's
                                               serial CUDA-graph path (fallback)
    YUE2_AR_CONCURRENCY     default 4          songs whose AR stages share one vLLM wave
    YUE2_NAR_BATCH_SIZE     default 2          songs per flow-matching batch (max 2)
    YUE2_VLLM_GPU_MEMORY_UTILIZATION  0.30     vLLM's share of the card (weights + KV)
    YUE2_MAX_QUEUE          default 12         queued + running tasks; full -> 503
    YUE2_MEMORY_BUDGET_GIB  default 28         the PyTorch process's VRAM cap
    YUE2_TASK_TIMEOUT_SECONDS  default 1200    per task, queue time excluded
    YUE2_MP3_BITRATE        default 320k       CBR bitrate for .mp3 outputs
    YUE2_WARMUP             default 1          run one tiny song before /ready
    YUE2_CACHE              <model dir>        where the derived vLLM AR checkpoint
                                               lives (falls back to /tmp when the
                                               model dir is read-only)
    VLLM_CACHE_ROOT         <model>/vllm-cache vLLM's torch.compile cache, same rule
    YUE2_VLLM_PYTHON / YUE2_VLLM_ANY_VERSION   set by the image; see src/yue2/fast.py

Scheduling is Turbo's JobWorker, subclassed only where the GPUStack contract
differs: results go to the facade's save_result_path (not an artifacts dir), a
truncated song fails instead of shipping, covers transcribe the facade's NFS
recording with our SheetSage2 loader, and errors carry our error types. Its
SQLite queue lives under YUE2_DATA_DIR in the container, so a restarted
instance starts empty and the facade's sweeper re-dispatches (status 404), the
same semantics as the in-memory queue this replaced.

Beside the audio at save_result_path the worker writes two sidecars into the
same directory, which the facade's janitor reaps with the audio (it rmtree's
whole day directories):
    <stem>.abc   the score the song was rendered from — planned by YuE2,
                 transcribed from a cover's source, or the one supplied (absent
                 for cot=off). Edit it and submit it back as ``abc`` to re-render.
    <stem>.json  seed, mode, execution path, token counts, timings, weights
"""
import importlib
import importlib.util
import json
import logging
import os
import secrets
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from harness.routes import REF_KEY, SAVE_KEY, ProgressBook, create_tasks_router
from yue2.service import JobWorker, Settings, build_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("yue2.server")

MODEL_DIR = os.environ.get("YUE2_MODEL_DIR", "/weights")
VAE_DIR = os.environ.get("YUE2_VAE_DIR") or None
SHEETSAGE2_DIR = os.environ.get("YUE2_SHEETSAGE2_DIR") or os.path.join(MODEL_DIR, "SheetSage2")
MERT_DIR = os.environ.get("YUE2_MERT_DIR") or os.path.join(MODEL_DIR, "MERT-v2-FullSong")
MP3_BITRATE = os.environ.get("YUE2_MP3_BITRATE", "320k")
SAMPLE_RATE = 48000


def _env_flag(name: str, default: str) -> bool:
    return os.environ.get(name, default) not in {"0", "false", "False", ""}


def make_settings(**overrides) -> Settings:
    """Turbo's Settings from our env names. The API key is Turbo's own HTTP auth,
    which this server does not expose; it only has to satisfy the model."""
    concurrency = int(os.environ.get("YUE2_AR_CONCURRENCY", "4"))
    values = dict(
        api_key=secrets.token_hex(16),
        data_dir=Path(os.environ.get("YUE2_DATA_DIR", "/tmp/yue2-jobs")),
        model=MODEL_DIR,
        # The default hub id is what makes from_pretrained honour pipeline.json's
        # decoder; an explicit dir overrides it.
        vae=VAE_DIR or "m-a-p/YuE2-Vae",
        backend=os.environ.get("YUE2_BACKEND", "vllm"),
        resident_models=True,
        memory_budget_gib=float(os.environ.get("YUE2_MEMORY_BUDGET_GIB", "28")),
        ar_concurrency=concurrency,
        vllm_max_num_seqs=concurrency,
        vllm_gpu_memory_utilization=float(os.environ.get("YUE2_VLLM_GPU_MEMORY_UTILIZATION", "0.30")),
        nar_batch_size=int(os.environ.get("YUE2_NAR_BATCH_SIZE", "2")),
        ar_nar_overlap=True,
        max_pending=int(os.environ.get("YUE2_MAX_QUEUE", "12")),
        task_timeout_seconds=float(os.environ.get("YUE2_TASK_TIMEOUT_SECONDS", "1200")),
        warmup=_env_flag("YUE2_WARMUP", "1"),
        local_files_only=True,
        # Turbo's own lazy SheetSage2 loader goes through trust_remote_code, which
        # transformers 4.57 breaks on local snapshots; covers use ours below.
        sheetsage_device="off",
    )
    values.update(overrides)
    return Settings(**values)


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


def _sidecars(save_result_path: str) -> list:
    root = os.path.splitext(save_result_path)[0]
    return [root + ".abc", root + ".json"]


def _publish(audio: np.ndarray, save_result_path: str, score: Optional[str], record: dict) -> None:
    """Write sidecars first, audio last, each via temp + os.replace.

    The audio's appearance at its final path is the completion signal; ordering
    it last means a reader that sees the song can always find its score."""
    os.makedirs(os.path.dirname(save_result_path) or ".", exist_ok=True)
    score_path, record_path = _sidecars(save_result_path)
    if score is not None:
        _atomic_text(score_path, score)
    _atomic_text(record_path, json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    tmp = _part_path(save_result_path)
    try:
        _write_audio(audio, tmp)
        os.replace(tmp, save_result_path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def _unpublish(save_result_path: str) -> None:
    for path in [save_result_path, *_sidecars(save_result_path)]:
        try:
            os.remove(path)
        except OSError:
            pass


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


_abc_tools = _load_abc_tools()


def load_transcriber():
    """SheetSage2 on the GPU, or None when the cover weights are not installed.

    Imported as a regular package from the snapshot directory rather than
    through AutoModel(trust_remote_code=True): transformers 4.57 copies only a
    model file's direct relative imports into its dynamic-module cache, and
    SheetSage2 has second-level ones, so the remote-code route dies on a missing
    chord_spelling_sheetsage2.py. Its pins (torch 2.8, transformers 4.45, numpy
    1.24) are the authors' environment, not requirements: it runs unchanged on
    this image's torch / transformers 4.57.6 / numpy 2.x (measured)."""
    if not (os.path.isdir(SHEETSAGE2_DIR) and os.path.isdir(MERT_DIR)):
        logger.info(f"Cover disabled: {SHEETSAGE2_DIR} or {MERT_DIR} not found")
        return None
    parent, package = os.path.split(os.path.abspath(SHEETSAGE2_DIR))
    if parent not in sys.path:
        sys.path.insert(0, parent)
    module = importlib.import_module(f"{package}.modeling_sheetsage2")
    # base_model_path keeps the MERT-v2 parent offline; the loader still checks
    # it against the sha256 pinned in SheetSage2's config.
    started = time.time()
    model = module.SheetSage2Model.from_pretrained(
        SHEETSAGE2_DIR, local_files_only=True, base_model_path=MERT_DIR,
    ).eval().to("cuda")
    logger.info(f"SheetSage2 loaded in {time.time() - started:.1f}s; cover enabled")
    return model


def transcribe(transcriber, path: str, keep_chords: bool) -> tuple:
    """Source recording -> ABC score, validated like upstream's transcribe helper."""
    try:
        result = transcriber.transcribe(path, melody_only=not keep_chords)
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


# ------------------------------------------------------------------- worker
class HarnessWorker(JobWorker):
    """Turbo's scheduler with the GPUStack contract's output and error rules."""

    def __init__(self, settings, factory, transcriber_loader=None):
        super().__init__(settings, factory)
        self._transcriber_loader = transcriber_loader
        self._transcriber_loaded = False
        self.cover_transcriber = None
        # MP3 encoding (single-threaded LAME, ~1-2 s per song on these ARM hosts)
        # and the NFS writes run off the scheduler thread. Done inline they
        # queued the rest of a 4-song wave behind each encode: system RTF 0.168
        # at 4-way against 0.137 for Turbo's own server on the same card.
        self._publisher = ThreadPoolExecutor(max_workers=max(1, settings.ar_concurrency),
                                             thread_name_prefix="yue2-publish")

    def stop(self):
        super().stop()
        # Let songs already rendered land on disk; GPUStack's stop grace period
        # bounds this, and the facade re-dispatches anything cut short.
        self._publisher.shutdown(wait=True)

    def _load(self):
        # Once per process, before the pipeline: vLLM checks for its share of
        # free memory at start, and SheetSage2 (~3.4 GiB) must already be counted.
        # _reload_pipeline() comes back through here and keeps the transcriber.
        if not self._transcriber_loaded and self._transcriber_loader is not None:
            self.cover_transcriber = self._transcriber_loader()
            self._transcriber_loaded = True
        super()._load()

    def _parallel_ready(self, context):
        # A cover transcribes first, so it runs on the serial path; its AR still
        # goes through the vLLM worker, just not batched with other songs.
        if context["request"].get(REF_KEY):
            return False
        return super()._parallel_ready(context)

    def _transcribe_cover(self, context):
        request = context["request"]
        path = request.get(REF_KEY)
        if not path:
            return
        if self.cover_transcriber is None:
            raise TranscriptionError("cover requested but SheetSage2 is not loaded")
        context["stage"]("transcribing")
        started = time.perf_counter()
        # cot=full keeps the source's harmony; melody (the default) drops the
        # chord symbols so the new style chooses its own.
        abc, warnings = transcribe(self.cover_transcriber, path, keep_chords=request.get("cot") == "full")
        request.pop(REF_KEY, None)
        request["abc"] = abc
        context["cover"] = {"source_audio": path, "transcription_warnings": warnings,
                            "transcribe_seconds": round(time.perf_counter() - started, 3)}
        if context["cancelled"]():
            raise InterruptedError("Cancelled after transcription")

    def _save_result(self, context, result):
        job_id = context["job"]["id"]
        save_result_path = context["request"][SAVE_KEY]
        context["stage"]("saving")
        truncated = result.truncated
        if truncated.get("abc"):
            raise TruncatedError("score planning hit its 4096-token cap; shorten the lyrics")
        if truncated.get("semantic"):
            raise TruncatedError(
                "song hit the 9000-token (~6 min) ceiling and would end mid-phrase; shorten the lyrics"
            )
        semantic = result.semantic
        request = semantic.plan.request
        audio_seconds = len(result.audio) / SAMPLE_RATE
        covered = "cover" in context
        planned = request.cot != "off" and not covered and context["request"].get("abc") is None
        record = {
            "engine": "yue2",
            "seed": request.seed,
            "cot": request.cot,
            "cfg_scale": request.guidance,
            "score_source": ("transcribed" if covered else "planned" if planned
                             else "provided" if request.abc is not None else "none"),
            "sample_rate": SAMPLE_RATE,
            "channels": int(result.audio.shape[1]),
            "audio_seconds": round(audio_seconds, 3),
            "abc_tokens": len(semantic.plan.abc_ids),
            "semantic_tokens": len(semantic.tokens),
            # Which road the AR took: "vllm" (accelerated) or "torch" (fallback for
            # cot=off / custom CFG, or YUE2_BACKEND=torch).
            "backend": semantic.timing.get("backend_actual", result.config.get("backend")),
            "timing": result.timing,
            "weights": {name: {file: entry["sha256"] for file, entry in identity["files"].items()}
                        for name, identity in result.weights.items()},
        }
        if covered:
            record["cover"] = context["cover"]
        if context["cancelled"]():
            raise InterruptedError("Cancelled before saving")
        # The task stays "running" at stage "saving" until the file is written,
        # so the facade never sees "completed" without the audio in place.
        self._publisher.submit(self._publish_and_finish, job_id, save_result_path,
                               result.audio, result.abc, record)

    def _publish_and_finish(self, job_id, save_result_path, audio, score, record):
        try:
            _publish(audio, save_result_path, score, record)
        except Exception as error:  # noqa: BLE001 - any write failure fails this task only
            logger.exception(f"Publishing failed: task={job_id}")
            _unpublish(save_result_path)
            self.store.finish(job_id, "failed", error={"code": "publish_failed", "message": str(error)[:500]})
            return
        # finish() turns this into "cancelled" if a DELETE arrived meanwhile;
        # then the files it just wrote must go.
        self.store.finish(job_id, "succeeded", result={
            "save_result_path": save_result_path, "audio_seconds": record["audio_seconds"],
            "abc_tokens": record["abc_tokens"], "semantic_tokens": record["semantic_tokens"],
            "backend": record["backend"],
        })
        if self.store.get(job_id)["status"] == "cancelled":
            _unpublish(save_result_path)
            return
        logger.info(f"Task {job_id}: {record['audio_seconds']:.1f}s audio via {record['backend']} "
                    f"(cot={record['cot']}, abc={record['abc_tokens']}, semantic={record['semantic_tokens']})")

    def _handle_error(self, context, error):
        job_id = context["job"]["id"]
        if isinstance(error, InterruptedError):
            if (context["cancel_event"].is_set() or self.stop_event.is_set()
                    or time.monotonic() >= context["deadline"]):
                # A DELETE, a shutdown or the task deadline: Turbo's handler
                # records cancelled / timeout.
                return super()._handle_error(context, error)
            # Turbo would call any InterruptedError a cancel; one nobody asked
            # for is a failure the facade must not report as "cancelled".
            logger.error(f"Task {job_id} interrupted without a cancel: {error}")
            self.store.finish(job_id, "failed", error={"code": "interrupted", "message": str(error)})
            return False
        if isinstance(error, (TruncatedError, TranscriptionError)):
            self.store.finish(job_id, "failed", error={"code": error.error_type, "message": str(error)})
            return False
        # Turbo's own handler hides the message behind "see worker logs"; the
        # facade shows error text to users, so keep what actually went wrong.
        logger.exception(f"Generation failed: task={job_id}")
        invalid = isinstance(error, (ValueError, TypeError))
        self.store.finish(job_id, "failed", error={
            "code": "invalid_generation" if invalid else "inference_failed",
            "message": str(error)[:500] or type(error).__name__})
        # Only an inference failure warrants rebuilding the pipeline.
        return not invalid


def _default_cache_dir() -> None:
    """Keep what vLLM builds at startup next to the weights, so it is built once
    per cluster rather than once per container start:

    - YUE2_CACHE: the AR checkpoint derived from the MoT (4.1 GB, ~1 min).
    - VLLM_CACHE_ROOT: vLLM's torch.compile / AOT cache (51 s on A100, keyed by
      model config and vLLM version, so a new image recompiles on its own).

    A read-only model mount cannot hold them (derivation takes a lock file beside
    its output), so fall back to the container's /tmp there."""
    writable = os.access(MODEL_DIR, os.W_OK)
    os.environ.setdefault("YUE2_CACHE", MODEL_DIR if writable else "/tmp/yue2-cache")
    os.environ.setdefault("VLLM_CACHE_ROOT",
                          os.path.join(MODEL_DIR, "vllm-cache") if writable else "/tmp/vllm-cache")


_book = ProgressBook()
_worker: Optional[HarnessWorker] = None


def _get_worker() -> Optional[HarnessWorker]:
    return _worker


@asynccontextmanager
async def _lifespan(_app):
    global _worker
    _default_cache_dir()
    _worker = HarnessWorker(make_settings(), build_pipeline, transcriber_loader=load_transcriber)
    # The worker thread loads the models and runs the warmup song; the HTTP side
    # answers /ready with 503 until it is done.
    _worker.start()
    yield
    _worker.stop()


app = FastAPI(title="YuE2 music task server", lifespan=_lifespan)
app.include_router(create_tasks_router(_get_worker, _book))


@app.get("/ready")
def ready():
    worker = _worker
    if worker is None or not worker.ready:
        status = "failed" if worker is not None and worker.startup_error else "loading"
        return JSONResponse(status_code=503, content={"status": status})
    return {"status": "ready", "sample_rate": SAMPLE_RATE,
            "cover": worker.cover_transcriber is not None,
            "backend": worker.settings.backend, "ar_concurrency": worker.settings.ar_concurrency}


@app.get("/health")
def health():
    worker = _worker
    return {"status": "ok" if worker is not None and worker.ready else "loading"}
