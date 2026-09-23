"""Contract tests for the GPUStack task harness (server.py + harness/).

These guard what the GPUStack facade depends on verbatim — status strings, 503
backpressure, 404 for unknown tasks, submit-time validation, real mid-run
cancellation, truncation surfacing as failure — plus the output files. They run
against a fake pipeline, so no GPU, weights or torch are needed; the upstream
CPU workflow installs no fastapi and skips this module.
"""
import json
import shutil
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import server  # noqa: E402
import harness.routes  # noqa: E402
from harness.routes import create_tasks_router  # noqa: E402
from harness.schemas import MAX_LYRIC_UNITS, lyric_units  # noqa: E402
from harness.task_manager import TaskManager, TaskStatus  # noqa: E402
from harness.worker import TaskWorker  # noqa: E402

SCORE = "X:1\nT:fake\nK:C\nV:Vocal\nCDEF|GABc|\n"
# Real native-dialect scores shipped with upstream, so the ABC check is exercised
# for real: melody.abc has no chord symbols, score.abc does.
EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
MELODY = (EXAMPLES / "melody.abc").read_text(encoding="utf-8")
FULL_SCORE = (EXAMPLES / "score.abc").read_text(encoding="utf-8")


class FakeTranscriber:
    """Stands in for SheetSage2: transcribe(path, melody_only) -> result dict."""

    def __init__(self):
        self.calls = []
        self.result = None  # override to simulate failures

    def transcribe(self, path, melody_only=False):
        self.calls.append({"path": path, "melody_only": melody_only})
        if self.result is not None:
            return self.result
        return {"abc": MELODY if melody_only else FULL_SCORE, "warnings": ["low vocal confidence in bar 3"]}


class FakePipe:
    """Stands in for YuE2Pipeline: same stage methods, same callbacks."""

    weights = {
        "mot": {"files": {"model.safetensors": {"sha256": "a" * 64, "bytes": 1}}, "config_sha256": "c"},
        "vae": {"files": {"model.safetensors": {"sha256": "b" * 64, "bytes": 1}}, "config_sha256": "d"},
    }

    def __init__(self):
        self.abc_tokens = 40
        self.semantic_tokens = 100
        self.plan_truncated = False
        self.semantic_truncated = False
        # When set, generate_semantic parks until cancelled() turns true.
        self.hold_semantic = False
        self.in_semantic = threading.Event()
        self.saw_cancel = False
        self.raise_interrupt = False
        self.calls = []
        self.requests = []

    def plan(self, request=None, cancelled=None, on_token=None):
        self.calls.append("plan")
        self.requests.append(request)
        if request.cot == "off":
            return SimpleNamespace(request=request, abc=None, abc_ids=[], truncated=False,
                                   timing={}, prefix=[1])
        if request.abc is not None:
            return SimpleNamespace(request=request, abc=request.abc, abc_ids=[1, 2, 3],
                                   truncated=False, timing={"seconds": 0.0}, prefix=[1])
        for token in range(self.abc_tokens):
            if cancelled is not None and cancelled():
                raise InterruptedError("Cancelled during abc")
            if on_token is not None:
                on_token("abc", token)
        return SimpleNamespace(request=request, abc=SCORE, abc_ids=list(range(self.abc_tokens)),
                               truncated=self.plan_truncated, timing={"seconds": 0.1}, prefix=[1])

    def generate_semantic(self, plan, sampling=None, cancelled=None, on_token=None):
        self.calls.append("semantic")
        if self.raise_interrupt:
            raise InterruptedError("not a cancel")
        self.in_semantic.set()
        if self.hold_semantic:
            deadline = time.monotonic() + 5
            while not (cancelled and cancelled()):
                if time.monotonic() > deadline:
                    raise AssertionError("cancel never reached the pipeline")
                time.sleep(0.01)
            self.saw_cancel = True
            raise InterruptedError("Cancelled during semantic")
        for token in range(self.semantic_tokens):
            if on_token is not None:
                on_token("semantic", token)
        return SimpleNamespace(plan=plan, tokens=list(range(self.semantic_tokens)),
                               truncated=self.semantic_truncated, timing={"seconds": 0.2})

    def synthesize(self, semantic, cancelled=None):
        self.calls.append("synthesize")
        return np.zeros((8, 64), dtype=np.float32)

    def decode(self, latents):
        self.calls.append("decode")
        t = np.arange(4800) / 48000
        tone = 0.1 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
        return np.stack([tone, tone], axis=1)


@pytest.fixture
def h(monkeypatch, tmp_path):
    tm = TaskManager(max_queue_size=4)
    pipe = FakePipe()
    transcriber = FakeTranscriber()
    # _generate reports progress through the module-level manager and reads the
    # module-level pipeline/transcriber, so point them at this test's instances.
    monkeypatch.setattr(server, "_task_manager", tm)
    monkeypatch.setattr(server, "_pipe", pipe)
    monkeypatch.setattr(server, "_transcriber", transcriber)
    # Source-length probing shells out to ffprobe; tests set the answer.
    duration = {"seconds": 60.0}
    monkeypatch.setattr(harness.routes, "probe_duration", lambda path: duration["seconds"])
    app = FastAPI()
    app.include_router(create_tasks_router(tm, cover_enabled=lambda: server._transcriber is not None))
    worker = TaskWorker(tm, server._generate)
    source = tmp_path / "source.mp3"
    source.write_bytes(b"never decoded: probe_duration is patched")
    ns = SimpleNamespace(client=TestClient(app), tm=tm, pipe=pipe, worker=worker, tmp=tmp_path,
                         transcriber=transcriber, duration=duration, source=str(source))
    yield ns
    tm.cancel_all_tasks()
    worker.stop()


def body(h, name="song.wav", **extra):
    payload = {"prompt": "Mandarin pop ballad, female vocal, piano", "lyrics": "[Verse]\n街灯\n[Chorus]\n晚安",
               "save_result_path": str(h.tmp / name)}
    payload.update(extra)
    return payload


def wait_status(h, task_id, wanted, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = h.client.get(f"/v1/tasks/{task_id}/status").json()
        if status["status"] in wanted:
            return status
        time.sleep(0.01)
    raise AssertionError(f"task {task_id} never reached {wanted}: {status}")


def submit(h, **kwargs):
    response = h.client.post("/v1/tasks/music/", json=body(h, **kwargs))
    assert response.status_code == 200, response.text
    return response.json()["task_id"]


# ------------------------------------------------------------------ contract
def test_status_strings_are_the_facade_vocabulary():
    # The facade's _ENGINE_STATE_MAP matches these verbatim; cancelled is double-L.
    assert [s.value for s in TaskStatus] == ["pending", "processing", "completed", "failed", "cancelled"]


def test_submit_response_shape(h):
    response = h.client.post("/v1/tasks/music/", json=body(h))
    assert response.status_code == 200
    payload = response.json()
    assert payload["task_status"] == "pending"
    assert payload["save_result_path"] == str(h.tmp / "song.wav")
    assert payload["task_id"]


def test_audio_alias_accepts_the_same_body(h):
    assert h.client.post("/v1/tasks/audio/", json=body(h)).status_code == 200


def test_facade_passthrough_keys_are_ignored(h):
    # new-api flattens metadata and the facade forwards everything it does not
    # own; ACE-Step-shaped keys must not turn into a 422.
    extra = {"model": "yue2", "user_id": "7", "audio_duration": 60, "bpm": 90, "instrumental": False}
    assert h.client.post("/v1/tasks/music/", json=body(h, **extra)).status_code == 200


def test_unknown_task_is_404_everywhere(h):
    # The sweeper reads status 404 as "engine lost it" and re-dispatches.
    assert h.client.get("/v1/tasks/nope/status").status_code == 404
    assert h.client.get("/v1/tasks/nope/result").status_code == 404
    assert h.client.delete("/v1/tasks/nope").json()["stop_status"] == "do_nothing"


def test_queue_status_is_not_captured_as_a_task_id(h):
    payload = h.client.get("/v1/tasks/queue/status").json()
    assert payload["queue_size"] == 4 and payload["queue_available"] == 4


def test_queue_full_is_503(h):
    # Worker not started: everything stays PENDING and fills the FIFO.
    for index in range(4):
        submit(h, name=f"q{index}.wav")
    response = h.client.post("/v1/tasks/music/", json=body(h, name="overflow.wav"))
    assert response.status_code == 503


@pytest.mark.parametrize("override, fragment", [
    ({"save_result_path": None}, "save_result_path is required"),
    ({"save_result_path": "/tmp/x.ogg"}, "must end in one of"),
    ({"prompt": "   "}, "prompt (style description) is required"),
    ({"cot": "sideways"}, "cot must be"),
    ({"cot": "off", "abc": SCORE}, "External ABC requires"),
    # repaint reaches the same door with task_type stripped; it must not
    # silently degrade into a fresh text-to-music song.
    ({"src_audio_path": "/nfs-rw/inputs/src.mp3"}, "repaint (src_audio) is not supported"),
])
def test_submit_time_validation_is_400(h, override, fragment):
    response = h.client.post("/v1/tasks/music/", json=body(h, **override))
    assert response.status_code == 400
    assert fragment in response.json()["detail"]


def test_over_budget_lyrics_are_rejected_before_the_gpu(h):
    lyrics = "我" * (MAX_LYRIC_UNITS + 1)
    response = h.client.post("/v1/tasks/music/", json=body(h, lyrics=lyrics))
    assert response.status_code == 400
    assert "lyrics too long" in response.json()["detail"]
    assert h.pipe.calls == []


def test_lyric_units_weigh_latin_below_cjk():
    assert lyric_units("我爱你") == 3
    assert lyric_units("love you") == pytest.approx(7 * 0.385)
    assert lyric_units("[Verse]\n") == pytest.approx(7 * 0.385)


def test_out_of_range_cfg_is_a_client_error(h):
    response = h.client.post("/v1/tasks/music/", json=body(h, cfg_scale=50))
    assert 400 <= response.status_code < 500


# ---------------------------------------------------------------- lifecycle
def test_completed_task_writes_audio_and_both_sidecars(h):
    import soundfile as sf

    h.worker.start()
    task_id = submit(h, seed=7)
    status = wait_status(h, task_id, {"completed", "failed"})
    assert status["status"] == "completed", status
    audio, rate = sf.read(h.tmp / "song.wav")
    assert rate == 48000 and audio.shape == (4800, 2)
    assert (h.tmp / "song.abc").read_text() == SCORE
    record = json.loads((h.tmp / "song.json").read_text())
    assert record["seed"] == 7
    assert record["cot"] == "full"
    assert record["score_source"] == "planned"
    assert (record["abc_tokens"], record["semantic_tokens"]) == (40, 100)
    assert record["weights"]["mot"] == {"model.safetensors": "a" * 64}
    # No temp file survives a successful publish.
    assert not list(h.tmp.glob("*.part*"))
    result = h.client.get(f"/v1/tasks/{task_id}/result")
    assert result.status_code == 200 and result.headers["content-type"] == "audio/wav"


def test_seed_is_pinned_at_submit_when_absent(h):
    h.worker.start()
    task_id = submit(h)
    wait_status(h, task_id, {"completed"})
    seed = json.loads((h.tmp / "song.json").read_text())["seed"]
    assert isinstance(seed, int) and 0 <= seed < 2**31


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_mp3_output_is_real_mp3(h):
    h.worker.start()
    task_id = submit(h, name="song.mp3")
    assert wait_status(h, task_id, {"completed", "failed"})["status"] == "completed"
    head = (h.tmp / "song.mp3").read_bytes()[:3]
    assert head == b"ID3" or head[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2")
    result = h.client.get(f"/v1/tasks/{task_id}/result")
    assert result.headers["content-type"] == "audio/mpeg"


def test_provided_score_skips_planning_and_is_echoed(h):
    h.worker.start()
    task_id = submit(h, abc=SCORE, cot="melody")
    assert wait_status(h, task_id, {"completed"})["status"] == "completed"
    assert (h.tmp / "song.abc").read_text() == SCORE
    assert json.loads((h.tmp / "song.json").read_text())["score_source"] == "provided"


def test_cot_off_writes_no_score_sidecar(h):
    h.worker.start()
    task_id = submit(h, cot="off")
    assert wait_status(h, task_id, {"completed"})["status"] == "completed"
    assert not (h.tmp / "song.abc").exists()
    assert json.loads((h.tmp / "song.json").read_text())["score_source"] == "none"


def test_progress_uses_facade_phases(h):
    h.pipe.hold_semantic = True
    h.worker.start()
    task_id = submit(h)
    assert h.pipe.in_semantic.wait(5)
    status = h.client.get(f"/v1/tasks/{task_id}/status").json()
    assert status["status"] == "processing"
    # Planning finished (40 tokens reported in steps of 16) inside "denoise",
    # as a global percentage within the score's 0-30 span.
    assert status["phase"] == "denoise"
    assert 0 < status["progress"] <= 30
    assert "phase_progress" not in status
    h.client.delete(f"/v1/tasks/{task_id}")


def test_progress_never_moves_backwards(h):
    task_id = submit(h)
    h.tm.set_progress(task_id, "denoise", 40)
    h.tm.set_progress(task_id, "denoise", 25)
    assert h.client.get(f"/v1/tasks/{task_id}/status").json()["progress"] == 40
    h.tm.set_progress(task_id, "decode")
    status = h.client.get(f"/v1/tasks/{task_id}/status").json()
    assert (status["phase"], status["progress"]) == ("decode", 40)
    h.tm.set_progress(task_id, "save", 150)
    assert h.client.get(f"/v1/tasks/{task_id}/status").json()["progress"] == 99


# -------------------------------------------------------------- cancellation
def test_cancel_pending_never_runs(h):
    task_id = submit(h)
    assert h.client.delete(f"/v1/tasks/{task_id}").json()["stop_status"] == "success"
    h.worker.start()
    time.sleep(0.2)
    assert h.client.get(f"/v1/tasks/{task_id}/status").json()["status"] == "cancelled"
    assert h.pipe.calls == []


def test_cancel_processing_stops_the_pipeline(h):
    h.pipe.hold_semantic = True
    h.worker.start()
    task_id = submit(h)
    assert h.pipe.in_semantic.wait(5)
    assert h.client.delete(f"/v1/tasks/{task_id}").json()["stop_status"] == "success"
    deadline = time.monotonic() + 5
    while not h.pipe.saw_cancel and time.monotonic() < deadline:
        time.sleep(0.01)
    assert h.pipe.saw_cancel, "the running stage never observed the cancel"
    time.sleep(0.1)
    status = h.client.get(f"/v1/tasks/{task_id}/status").json()
    assert status["status"] == "cancelled"
    assert "synthesize" not in h.pipe.calls
    assert not (h.tmp / "song.wav").exists()
    # The worker is free again: the next task runs.
    h.pipe.hold_semantic = False
    next_id = submit(h, name="next.wav")
    assert wait_status(h, next_id, {"completed", "failed"})["status"] == "completed"


def test_an_interrupt_that_is_not_a_cancel_fails_the_task(h):
    h.pipe.raise_interrupt = True
    h.worker.start()
    task_id = submit(h)
    status = wait_status(h, task_id, {"failed", "cancelled", "completed"})
    assert status["status"] == "failed"


# --------------------------------------------------------------- truncation
def test_truncated_song_fails_instead_of_shipping(h):
    h.pipe.semantic_truncated = True
    h.worker.start()
    task_id = submit(h)
    status = wait_status(h, task_id, {"failed", "completed"})
    assert status["status"] == "failed"
    assert status["error_type"] == "truncated"
    assert not (h.tmp / "song.wav").exists()


def test_truncated_score_fails_before_rendering(h):
    h.pipe.plan_truncated = True
    h.worker.start()
    task_id = submit(h)
    status = wait_status(h, task_id, {"failed", "completed"})
    assert status["error_type"] == "truncated"
    assert "semantic" not in h.pipe.calls


# -------------------------------------------------------------------- cover
def test_cover_renders_the_transcribed_melody(h):
    h.worker.start()
    task_id = submit(h, reference_audio_path=h.source, lyrics="[Verse]\n换一种唱法")
    status = wait_status(h, task_id, {"completed", "failed"})
    assert status["status"] == "completed", status
    # Default cover mode is melody-only: chords stripped at transcription...
    assert h.transcriber.calls == [{"path": h.source, "melody_only": True}]
    # ...and that exact score is what YuE2 rendered, with no planning stage.
    rendered = h.pipe.requests[-1]
    assert (rendered.cot, rendered.abc) == ("melody", MELODY)
    assert (h.tmp / "song.abc").read_text() == MELODY
    record = json.loads((h.tmp / "song.json").read_text())
    assert record["score_source"] == "transcribed"
    assert record["cover"]["source_audio"] == h.source
    assert record["cover"]["transcription_warnings"] == ["low vocal confidence in bar 3"]


def test_cover_full_keeps_the_source_harmony(h):
    h.worker.start()
    task_id = submit(h, reference_audio_path=h.source, cot="full")
    assert wait_status(h, task_id, {"completed", "failed"})["status"] == "completed"
    assert h.transcriber.calls[-1]["melody_only"] is False
    assert h.pipe.requests[-1].abc == FULL_SCORE


def test_cover_is_refused_when_sheetsage2_is_not_installed(h, monkeypatch):
    monkeypatch.setattr(server, "_transcriber", None)
    response = h.client.post("/v1/tasks/music/", json=body(h, reference_audio_path=h.source))
    assert response.status_code == 400
    assert "cover is not available" in response.json()["detail"]


@pytest.mark.parametrize("override, fragment", [
    ({"reference_audio_path": "/nfs-rw/inputs/missing.mp3"}, "does not exist"),
    ({"abc": SCORE}, "do not also send abc"),
    ({"cot": "off"}, "cot must be melody or full"),
])
def test_cover_submit_time_validation(h, override, fragment):
    payload = body(h, reference_audio_path=h.source)
    payload.update(override)
    response = h.client.post("/v1/tasks/music/", json=payload)
    assert response.status_code == 400
    assert fragment in response.json()["detail"]
    assert h.transcriber.calls == []


def test_cover_source_over_the_length_cap_is_rejected(h):
    h.duration["seconds"] = 400.0
    response = h.client.post("/v1/tasks/music/", json=body(h, reference_audio_path=h.source))
    assert response.status_code == 400
    assert "reference audio too long" in response.json()["detail"]


def test_undecodable_cover_source_is_rejected(h):
    h.duration["seconds"] = None
    response = h.client.post("/v1/tasks/music/", json=body(h, reference_audio_path=h.source))
    assert response.status_code == 400
    assert "cannot be decoded" in response.json()["detail"]


def test_transcription_without_a_score_fails_the_task(h):
    h.transcriber.result = {"abc": None, "abc_error": "no melody detected"}
    h.worker.start()
    task_id = submit(h, reference_audio_path=h.source)
    status = wait_status(h, task_id, {"failed", "completed"})
    assert (status["status"], status["error_type"]) == ("failed", "transcription")
    assert h.pipe.calls == []


def test_chords_in_a_melody_transcription_fail_the_task(h):
    h.transcriber.result = {"abc": FULL_SCORE}
    h.worker.start()
    task_id = submit(h, reference_audio_path=h.source)
    status = wait_status(h, task_id, {"failed", "completed"})
    assert (status["status"], status["error_type"]) == ("failed", "transcription")


# ------------------------------------------------------------------ health
def test_ready_is_503_until_the_pipeline_is_loaded(monkeypatch):
    monkeypatch.setattr(server, "_pipe", None)
    assert server.ready().status_code == 503
    monkeypatch.setattr(server, "_pipe", FakePipe())
    assert server.ready()["status"] == "ready"
