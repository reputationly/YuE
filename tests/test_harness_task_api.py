"""Contract tests for the GPUStack task harness (server.py + harness/).

These guard what the GPUStack facade depends on verbatim — status strings, 503
backpressure, 404 for unknown tasks, submit-time validation, real mid-run
cancellation, truncation surfacing as failure — plus the output files and the
per-request routing between Turbo's accelerated (parallel) and fallback
(serial) paths. They drive the real Turbo JobWorker with a fake pipeline, so
no GPU, weights or vLLM are needed.
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

import harness.routes  # noqa: E402
import server  # noqa: E402
from harness.routes import FACADE_STATUS, ProgressBook, create_tasks_router  # noqa: E402
from harness.schemas import MAX_LYRIC_UNITS, lyric_units  # noqa: E402

SCORE = "X:1\nT:fake\nK:C\nV:Vocal\nCDEF|GABc|\n"
# Real native-dialect scores shipped with upstream, so the ABC check is exercised
# for real: melody.abc has no chord symbols, score.abc does.
EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
MELODY = (EXAMPLES / "melody.abc").read_text(encoding="utf-8")
FULL_SCORE = (EXAMPLES / "score.abc").read_text(encoding="utf-8")
TERMINAL = {"completed", "failed", "cancelled"}


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
    """Implements both roads Turbo's JobWorker drives: the accelerated one
    (generate_ar -> generate_nar_batch / render_ar -> render_nar) and the serial
    fallback (__call__). Records which road each song took."""

    def __init__(self):
        self.abc_tokens = 40
        self.semantic_tokens = 100
        self.plan_truncated = False
        self.semantic_truncated = False
        self.raise_interrupt = False
        # While hold is set, the semantic stage parks until released or cancelled.
        self.hold = threading.Event()
        self.in_semantic = threading.Event()
        self.saw_cancel = threading.Event()
        self.lock = threading.Lock()
        self.paths = []      # ("parallel" | "serial", seed)
        self.requests = []   # generation fields as the pipeline received them

    def preload(self):
        pass

    def close(self):
        self.hold.clear()

    def parallel_ar_eligible(self, request):
        return request.get("cot", "full") in {"full", "melody"} and request.get("cfg_scale") in (None, 1)

    def __call__(self, *, cancelled=None, on_token=None, on_stage=None, **fields):
        ar = self._ar("serial", fields, cancelled, on_token, on_stage)
        return self.render_ar(ar, cancelled=cancelled, on_stage=on_stage)

    def generate_ar(self, *, cancelled=None, on_token=None, on_stage=None, **fields):
        return self._ar("parallel", fields, cancelled, on_token, on_stage)

    def _ar(self, path, fields, cancelled, on_token, on_stage):
        with self.lock:
            self.paths.append((path, fields.get("seed")))
            self.requests.append(dict(fields))
        if self.raise_interrupt:
            raise InterruptedError("not a cancel")
        if on_stage:
            on_stage("planning")
        planned = fields.get("cot", "full") != "off" and fields.get("abc") is None
        if planned:
            for token in range(self.abc_tokens):
                if cancelled and cancelled():
                    raise InterruptedError("Cancelled during abc")
                if on_token:
                    on_token("abc", token)
        if on_stage:
            on_stage("semantic")
        self.in_semantic.set()
        deadline = time.monotonic() + 10
        while self.hold.is_set():
            if cancelled and cancelled():
                self.saw_cancel.set()
                raise InterruptedError("Cancelled during semantic")
            if time.monotonic() > deadline:
                raise AssertionError("hold never released")
            time.sleep(0.01)
        for token in range(self.semantic_tokens):
            if cancelled and cancelled():
                raise InterruptedError("Cancelled during semantic")
            if on_token:
                on_token("semantic", token)
        return SimpleNamespace(fields=dict(fields), planned=planned, path=path)

    def nar_batch_admission(self, results):
        return {"allowed": True}

    def generate_nar_batch(self, results, *, cancelled=None, on_stage=None):
        for stage in on_stage or []:
            stage("synthesis")
        return list(results)

    def render_ar(self, ar, *, cancelled=None, on_stage=None):
        if on_stage:
            on_stage("synthesis")
        return self.render_nar(ar, cancelled=cancelled, on_stage=on_stage)

    def render_nar(self, ar, *, cancelled=None, on_stage=None):
        if on_stage:
            on_stage("decode")
        fields = ar.fields
        cfg = fields.get("cfg_scale")
        request = SimpleNamespace(seed=fields.get("seed"), cot=fields.get("cot", "full"),
                                  abc=fields.get("abc"), guidance=1.0 if cfg is None else cfg)
        plan = SimpleNamespace(request=request,
                               abc_ids=list(range(self.abc_tokens if ar.planned else (3 if fields.get("abc") else 0))))
        semantic = SimpleNamespace(plan=plan, tokens=list(range(self.semantic_tokens)),
                                   timing={"backend_actual": "vllm" if ar.path == "parallel" else "torch"})
        t = np.arange(4800) / 48000
        tone = (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        return SimpleNamespace(
            audio=np.stack([tone, tone], axis=1),
            abc=SCORE if ar.planned else fields.get("abc"),
            truncated={"abc": self.plan_truncated, "semantic": self.semantic_truncated},
            semantic=semantic, timing={"e2e_seconds": 0.01}, config={"backend": "vllm"},
            weights={"mot": {"files": {"model.safetensors": {"sha256": "a" * 64, "bytes": 1}}},
                     "vae": {"files": {"model.safetensors": {"sha256": "b" * 64, "bytes": 1}}}},
        )


def wait_for(check, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = check()
        if value:
            return value
        time.sleep(0.01)
    raise AssertionError("timed out waiting")


@pytest.fixture
def h(monkeypatch, tmp_path):
    pipe, transcriber = FakePipe(), FakeTranscriber()
    ns = SimpleNamespace(pipe=pipe, transcriber=transcriber, tmp=tmp_path, duration={"seconds": 60.0},
                         book=ProgressBook(), worker=None)
    # Source-length probing shells out to ffprobe; tests set the answer.
    monkeypatch.setattr(harness.routes, "probe_duration", lambda path: ns.duration["seconds"])
    source = tmp_path / "source.mp3"
    source.write_bytes(b"never decoded: probe_duration is patched")
    ns.source = str(source)

    def start(max_pending=4, cover=True, **overrides):
        settings = server.make_settings(data_dir=tmp_path / "jobs", warmup=False, ar_batch_wait_ms=0,
                                        max_pending=max_pending, **overrides)
        worker = server.HarnessWorker(settings, lambda _settings: pipe,
                                      transcriber_loader=(lambda: transcriber) if cover else None)
        worker.start()
        wait_for(lambda: worker.ready)
        app = FastAPI()
        app.include_router(create_tasks_router(lambda: worker, ns.book))
        ns.worker, ns.client = worker, TestClient(app)
        return ns

    ns.start = start
    yield ns
    pipe.hold.clear()
    if ns.worker is not None:
        ns.worker.stop()


def body(h, name="song.wav", **extra):
    payload = {"prompt": "Mandarin pop ballad, female vocal, piano", "lyrics": "[Verse]\n街灯\n[Chorus]\n晚安",
               "save_result_path": str(h.tmp / name)}
    payload.update(extra)
    return payload


def submit(h, **kwargs):
    response = h.client.post("/v1/tasks/music/", json=body(h, **kwargs))
    assert response.status_code == 200, response.text
    return response.json()["task_id"]


def status_of(h, task_id):
    return h.client.get(f"/v1/tasks/{task_id}/status").json()


def wait_status(h, task_id, wanted=TERMINAL, timeout=5.0):
    return wait_for(lambda: (lambda s: s if s["status"] in wanted else None)(status_of(h, task_id)), timeout)


# ------------------------------------------------------------------ contract
def test_status_strings_are_the_facade_vocabulary():
    # The facade's _ENGINE_STATE_MAP matches these verbatim; cancelled is double-L.
    assert set(FACADE_STATUS.values()) == {"pending", "processing", "completed", "failed", "cancelled"}


def test_submit_response_shape(h):
    h.start()
    response = h.client.post("/v1/tasks/music/", json=body(h))
    assert response.status_code == 200
    payload = response.json()
    assert payload["task_status"] == "pending"
    assert payload["save_result_path"] == str(h.tmp / "song.wav")
    assert payload["task_id"]


def test_audio_alias_accepts_the_same_body(h):
    h.start()
    assert h.client.post("/v1/tasks/audio/", json=body(h)).status_code == 200


def test_facade_passthrough_keys_are_ignored(h):
    # new-api flattens metadata and the facade forwards everything it does not
    # own; ACE-Step-shaped keys must not turn into a 422.
    h.start()
    extra = {"model": "yue2", "user_id": "7", "audio_duration": 60, "bpm": 90, "instrumental": False}
    assert h.client.post("/v1/tasks/music/", json=body(h, **extra)).status_code == 200


def test_unknown_task_is_404_everywhere(h):
    # The sweeper reads status 404 as "engine lost it" and re-dispatches.
    h.start()
    assert h.client.get("/v1/tasks/nope/status").status_code == 404
    assert h.client.get("/v1/tasks/nope/result").status_code == 404
    assert h.client.delete("/v1/tasks/nope").json()["stop_status"] == "do_nothing"


def test_queue_status_is_not_captured_as_a_task_id(h):
    h.start()
    payload = h.client.get("/v1/tasks/queue/status").json()
    assert payload["queue_size"] == 4 and payload["queue_available"] == 4


def test_queue_full_is_503(h):
    h.pipe.hold.set()
    h.start(max_pending=4)
    for index in range(4):
        submit(h, name=f"q{index}.wav")
    response = h.client.post("/v1/tasks/music/", json=body(h, name="overflow.wav"))
    assert response.status_code == 503


def test_submissions_before_ready_are_503(h):
    app = FastAPI()
    app.include_router(create_tasks_router(lambda: None, h.book))
    response = TestClient(app).post("/v1/tasks/music/", json=body(h))
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
    h.start()
    response = h.client.post("/v1/tasks/music/", json=body(h, **override))
    assert response.status_code == 400
    assert fragment in response.json()["detail"]


def test_over_budget_lyrics_are_rejected_before_the_gpu(h):
    h.start()
    response = h.client.post("/v1/tasks/music/", json=body(h, lyrics="我" * (MAX_LYRIC_UNITS + 1)))
    assert response.status_code == 400
    assert "lyrics too long" in response.json()["detail"]
    assert h.pipe.paths == []


def test_lyric_units_weigh_latin_below_cjk():
    assert lyric_units("我爱你") == 3
    assert lyric_units("love you") == pytest.approx(7 * 0.385)
    assert lyric_units("[Verse]\n") == pytest.approx(7 * 0.385)


def test_out_of_range_cfg_is_a_client_error(h):
    h.start()
    response = h.client.post("/v1/tasks/music/", json=body(h, cfg_scale=50))
    assert 400 <= response.status_code < 500


# ---------------------------------------------------------------- lifecycle
def test_completed_task_writes_audio_and_both_sidecars(h):
    import soundfile as sf

    h.start()
    task_id = submit(h, seed=7)
    status = wait_status(h, task_id)
    assert status["status"] == "completed", status
    audio, rate = sf.read(h.tmp / "song.wav")
    assert rate == 48000 and audio.shape == (4800, 2)
    assert (h.tmp / "song.abc").read_text() == SCORE
    record = json.loads((h.tmp / "song.json").read_text())
    assert record["seed"] == 7
    assert record["cot"] == "full"
    assert record["score_source"] == "planned"
    assert record["backend"] == "vllm"
    assert (record["abc_tokens"], record["semantic_tokens"]) == (40, 100)
    assert record["weights"]["mot"] == {"model.safetensors": "a" * 64}
    assert not list(h.tmp.glob("*.part*"))
    result = h.client.get(f"/v1/tasks/{task_id}/result")
    assert result.status_code == 200 and result.headers["content-type"] == "audio/wav"


def test_seed_is_pinned_at_submit_when_absent(h):
    h.start()
    task_id = submit(h)
    wait_status(h, task_id)
    seed = json.loads((h.tmp / "song.json").read_text())["seed"]
    assert isinstance(seed, int) and 0 <= seed < 2**31


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_mp3_output_is_real_mp3(h):
    h.start()
    task_id = submit(h, name="song.mp3")
    assert wait_status(h, task_id)["status"] == "completed"
    head = (h.tmp / "song.mp3").read_bytes()[:3]
    assert head == b"ID3" or head[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2")
    result = h.client.get(f"/v1/tasks/{task_id}/result")
    assert result.headers["content-type"] == "audio/mpeg"


def test_a_write_failure_fails_the_task_instead_of_hanging_in_saving(h):
    # Encoding and NFS writes run off the scheduler thread; a failure there must
    # still reach the task, not leave it "processing" at stage "saving" forever.
    h.start()
    (h.tmp / "blocker").write_text("a file where the output directory should be")
    task_id = submit(h, name="blocker/song.wav")
    status = wait_status(h, task_id)
    assert (status["status"], status["error_type"]) == ("failed", "publish_failed")


def test_provided_score_is_rendered_and_echoed(h):
    h.start()
    task_id = submit(h, abc=SCORE, cot="melody")
    assert wait_status(h, task_id)["status"] == "completed"
    assert h.pipe.requests[-1]["abc"] == SCORE
    assert (h.tmp / "song.abc").read_text() == SCORE
    assert json.loads((h.tmp / "song.json").read_text())["score_source"] == "provided"


# ------------------------------------------------------------------ routing
def test_concurrent_songs_share_the_accelerated_path(h):
    h.start(max_pending=8)
    ids = [submit(h, name=f"c{i}.wav", seed=i) for i in range(4)]
    for task_id in ids:
        assert wait_status(h, task_id)["status"] == "completed"
    assert sorted(h.pipe.paths) == [("parallel", i) for i in range(4)]


@pytest.mark.parametrize("override", [{"cot": "off"}, {"cfg_scale": 3.0}])
def test_requests_vllm_cannot_serve_fall_back_to_the_serial_path(h, override):
    h.start()
    task_id = submit(h, seed=11, **override)
    assert wait_status(h, task_id)["status"] == "completed"
    assert h.pipe.paths == [("serial", 11)]
    assert json.loads((h.tmp / "song.json").read_text())["backend"] == "torch"


def test_cot_off_writes_no_score_sidecar(h):
    h.start()
    task_id = submit(h, cot="off")
    assert wait_status(h, task_id)["status"] == "completed"
    assert not (h.tmp / "song.abc").exists()
    assert json.loads((h.tmp / "song.json").read_text())["score_source"] == "none"


# ------------------------------------------------------------------ progress
def test_progress_uses_facade_phases(h):
    h.pipe.hold.set()
    h.start()
    task_id = submit(h)
    assert h.pipe.in_semantic.wait(5)
    # Turbo writes token counts about once a second; wait for the fold to see them.
    status = wait_for(lambda: (lambda s: s if s.get("progress") else None)(status_of(h, task_id)))
    assert status["status"] == "processing"
    assert status["phase"] == "denoise"
    # Planning is done (40 of an expected 400 score tokens) and semantic has not
    # started, so the value sits inside the score's 0-30 span.
    assert 0 < status["progress"] <= 30
    assert "phase_progress" not in status


def test_progress_book_never_moves_backwards():
    book = ProgressBook()
    book.register("t", units=10, planned=True, cover=False, save_result_path="/x.mp3")
    assert book.fold("t", {"stage": "synthesis", "tokens": {}}) == ("denoise", 80.0)
    # A late token report from an earlier stage must not pull the bar back.
    assert book.fold("t", {"stage": "semantic", "tokens": {"abc": 100, "semantic": 1}}) == ("denoise", 80.0)
    assert book.fold("t", {"stage": "saving", "tokens": {}}) == ("save", 97.0)


def test_cover_semantic_progress_starts_after_transcription():
    book = ProgressBook()
    book.register("c", units=10, planned=False, cover=True, save_result_path="/x.mp3")
    assert book.fold("c", {"stage": "transcribing", "tokens": {}}) == ("encode", None)
    phase, value = book.fold("c", {"stage": "semantic", "tokens": {"abc": 0, "semantic": 1}})
    assert phase == "denoise" and 20 <= value < 21


# -------------------------------------------------------------- cancellation
def test_cancel_pending_never_runs(h):
    h.pipe.hold.set()
    h.start()
    running = submit(h, name="first.wav", seed=1)
    assert h.pipe.in_semantic.wait(5)
    # The worker claims the next wave only after this one's AR is done, so a
    # task submitted now stays queued.
    queued = submit(h, name="late.wav", seed=99)
    assert status_of(h, queued)["status"] == "pending"
    assert h.client.delete(f"/v1/tasks/{queued}").json()["stop_status"] == "success"
    assert status_of(h, queued)["status"] == "cancelled"
    h.pipe.hold.clear()
    assert wait_status(h, running)["status"] == "completed"
    time.sleep(0.2)
    assert 99 not in [seed for _, seed in h.pipe.paths]
    assert not (h.tmp / "late.wav").exists()


def test_cancel_processing_stops_the_pipeline(h):
    h.pipe.hold.set()
    h.start()
    task_id = submit(h)
    assert h.pipe.in_semantic.wait(5)
    assert h.client.delete(f"/v1/tasks/{task_id}").json()["stop_status"] == "success"
    # The caller sees the cancel at once, before the worker has noticed.
    assert status_of(h, task_id)["status"] == "cancelled"
    assert h.pipe.saw_cancel.wait(5), "the running stage never observed the cancel"
    wait_for(lambda: h.worker.store.get(task_id)["status"] == "cancelled")
    assert not (h.tmp / "song.wav").exists()
    # The worker is free again: the next task runs.
    h.pipe.hold.clear()
    next_id = submit(h, name="next.wav")
    assert wait_status(h, next_id)["status"] == "completed"


def test_an_interrupt_that_is_not_a_cancel_fails_the_task(h):
    h.pipe.raise_interrupt = True
    h.start()
    task_id = submit(h)
    status = wait_status(h, task_id)
    assert (status["status"], status["error_type"]) == ("failed", "interrupted")


def test_a_task_past_its_deadline_fails_as_timeout(h):
    h.pipe.hold.set()
    h.start(task_timeout_seconds=0.3)
    task_id = submit(h)
    status = wait_status(h, task_id)
    assert (status["status"], status["error_type"]) == ("failed", "timeout")


# --------------------------------------------------------------- truncation
def test_truncated_song_fails_instead_of_shipping(h):
    h.pipe.semantic_truncated = True
    h.start()
    task_id = submit(h)
    status = wait_status(h, task_id)
    assert (status["status"], status["error_type"]) == ("failed", "truncated")
    assert not (h.tmp / "song.wav").exists()


def test_truncated_score_fails(h):
    h.pipe.plan_truncated = True
    h.start()
    task_id = submit(h)
    status = wait_status(h, task_id)
    assert (status["status"], status["error_type"]) == ("failed", "truncated")
    assert "score planning" in status["error"]


# -------------------------------------------------------------------- cover
def test_cover_renders_the_transcribed_melody(h):
    h.start()
    task_id = submit(h, reference_audio_path=h.source, lyrics="[Verse]\n换一种唱法")
    status = wait_status(h, task_id)
    assert status["status"] == "completed", status
    # Default cover mode is melody-only: chords stripped at transcription...
    assert h.transcriber.calls == [{"path": h.source, "melody_only": True}]
    # ...and that exact score is what YuE2 rendered. A cover transcribes first,
    # so it takes the serial road (its AR still runs on vLLM, unbatched).
    assert h.pipe.requests[-1]["abc"] == MELODY and h.pipe.requests[-1]["cot"] == "melody"
    assert h.pipe.paths[-1][0] == "serial"
    assert (h.tmp / "song.abc").read_text() == MELODY
    record = json.loads((h.tmp / "song.json").read_text())
    assert record["score_source"] == "transcribed"
    assert record["cover"]["source_audio"] == h.source
    assert record["cover"]["transcription_warnings"] == ["low vocal confidence in bar 3"]


def test_cover_full_keeps_the_source_harmony(h):
    h.start()
    task_id = submit(h, reference_audio_path=h.source, cot="full")
    assert wait_status(h, task_id)["status"] == "completed"
    assert h.transcriber.calls[-1]["melody_only"] is False
    assert h.pipe.requests[-1]["abc"] == FULL_SCORE


def test_cover_is_refused_when_sheetsage2_is_not_installed(h):
    h.start(cover=False)
    response = h.client.post("/v1/tasks/music/", json=body(h, reference_audio_path=h.source))
    assert response.status_code == 400
    assert "cover is not available" in response.json()["detail"]


@pytest.mark.parametrize("override, fragment", [
    ({"reference_audio_path": "/nfs-rw/inputs/missing.mp3"}, "does not exist"),
    ({"abc": SCORE}, "do not also send abc"),
    ({"cot": "off"}, "cot must be melody or full"),
])
def test_cover_submit_time_validation(h, override, fragment):
    h.start()
    payload = body(h, reference_audio_path=h.source)
    payload.update(override)
    response = h.client.post("/v1/tasks/music/", json=payload)
    assert response.status_code == 400
    assert fragment in response.json()["detail"]
    assert h.transcriber.calls == []


def test_cover_source_over_the_length_cap_is_rejected(h):
    h.start()
    h.duration["seconds"] = 400.0
    response = h.client.post("/v1/tasks/music/", json=body(h, reference_audio_path=h.source))
    assert response.status_code == 400
    assert "reference audio too long" in response.json()["detail"]


def test_undecodable_cover_source_is_rejected(h):
    h.start()
    h.duration["seconds"] = None
    response = h.client.post("/v1/tasks/music/", json=body(h, reference_audio_path=h.source))
    assert response.status_code == 400
    assert "cannot be decoded" in response.json()["detail"]


def test_transcription_without_a_score_fails_the_task(h):
    h.transcriber.result = {"abc": None, "abc_error": "no melody detected"}
    h.start()
    task_id = submit(h, reference_audio_path=h.source)
    status = wait_status(h, task_id)
    assert (status["status"], status["error_type"]) == ("failed", "transcription")
    assert h.pipe.paths == []


def test_chords_in_a_melody_transcription_fail_the_task(h):
    h.transcriber.result = {"abc": FULL_SCORE}
    h.start()
    task_id = submit(h, reference_audio_path=h.source)
    status = wait_status(h, task_id)
    assert (status["status"], status["error_type"]) == ("failed", "transcription")


# ------------------------------------------------------------------ health
def test_ready_is_503_until_the_worker_is_loaded(monkeypatch):
    monkeypatch.setattr(server, "_worker", None)
    assert server.ready().status_code == 503
    monkeypatch.setattr(server, "_worker", SimpleNamespace(ready=False, startup_error=True))
    response = server.ready()
    assert response.status_code == 503 and b"failed" in response.body
