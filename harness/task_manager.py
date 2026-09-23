"""In-memory FIFO task manager for the YuE2 async task server.

Ported from reputationly/breeze-tts (itself a mirror of LightX2V's
``lightx2v/server/task_manager.py``) so the GPUStack facade can
dispatch/poll/cancel YuE2 exactly like every other built-in engine:

- OrderedDict FIFO, capacity = ``max_queue_size``; create over capacity -> raise
  (the route maps it to HTTP 503, which the facade reads as backpressure).
- Single worker thread processes PENDING -> PROCESSING -> COMPLETED/FAILED
  serially: YuE2's stages each hold the whole GPU (the pipeline moves the
  backbone to CPU for the VAE and back), so there is nothing to overlap.
- task_id is in-memory only; on instance restart tasks are lost -> the facade
  sweeper re-dispatches (status endpoint returning 404 triggers requeue).

Status strings MUST stay exactly ``pending/processing/completed/failed/cancelled``
— the facade's ``_ENGINE_STATE_MAP`` matches them verbatim (cancelled, double L).

One semantic difference from Breeze: cancelling a PROCESSING task really stops
it. Every YuE2 stage polls a ``cancelled()`` callback (per AR token, per
flow-matching step) and raises InterruptedError, so the worker hands the task's
``stop_event`` straight to the pipeline instead of running to completion and
discarding the result.
"""
import logging
import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class QueueFullError(RuntimeError):
    """Raised by create_task when the per-instance FIFO is at capacity.

    The route maps ONLY this to HTTP 503 (backpressure, retryable); other
    create-time errors (e.g. duplicate task id) are client errors -> 400.
    """


class TaskStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


TERMINAL_STATES = {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED}


@dataclass
class TaskInfo:
    task_id: str
    status: TaskStatus
    message: Any  # the MusicTaskRequest
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    error: Optional[str] = None
    error_type: Optional[str] = None
    save_result_path: Optional[str] = None
    stop_event: threading.Event = field(default_factory=threading.Event)
    # Progress reporting. The facade's contract takes either phase +
    # phase_progress (folded with its own per-kind stage weights) or a global
    # ``progress`` the engine folded itself, and the latter wins outright
    # (gpustack/server/video_progress.py). YuE2 reports the global value: its
    # stage costs (score ~30 %, semantic ~50 %, flow matching ~12 %, VAE ~6 %)
    # do not match the music kind's table (denoise 70 / decode 10), and a global
    # number never has to be re-based when the phase changes. ``phase`` rides
    # along as a label. Left None until the worker sets them; a task that never
    # reports falls back to the facade's elapsed-time estimate.
    phase: Optional[str] = None
    progress: Optional[float] = None


class TaskManager:
    def __init__(self, max_queue_size: int = 8):
        self.max_queue_size = max_queue_size

        self._tasks: "OrderedDict[str, TaskInfo]" = OrderedDict()
        self._lock = threading.RLock()
        self._task_available = threading.Condition(self._lock)

        self._processing_lock = threading.Lock()
        self._current_processing_task: Optional[str] = None

        self.total_tasks = 0
        self.completed_tasks = 0
        self.failed_tasks = 0

    # ------------------------------------------------------------------ create
    def create_task(self, message: Any) -> str:
        with self._task_available:
            existing_id = getattr(message, "task_id", None)
            if existing_id and existing_id in self._tasks:
                raise RuntimeError(f"Task ID {existing_id} already exists")

            active = sum(
                1
                for t in self._tasks.values()
                if t.status in (TaskStatus.PENDING, TaskStatus.PROCESSING)
            )
            if active >= self.max_queue_size:
                raise QueueFullError(
                    f"Task queue is full (max {self.max_queue_size} tasks)"
                )

            task_id = existing_id or uuid.uuid4().hex
            self._tasks[task_id] = TaskInfo(
                task_id=task_id,
                status=TaskStatus.PENDING,
                message=message,
                save_result_path=getattr(message, "save_result_path", None),
            )
            self.total_tasks += 1
            self._cleanup_old_tasks()
            self._task_available.notify()
            return task_id

    # ---------------------------------------------------------------- lifecycle
    def start_task(self, task_id: str) -> bool:
        """Atomically transition PENDING -> PROCESSING under the manager lock.

        Returns False if the task is missing or no longer PENDING — in
        particular when a DELETE cancelled it between the worker's pending
        check and this call. An unconditional overwrite here would resurrect a
        CANCELLED task as PROCESSING and leave it stuck forever.
        """
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None or task.status != TaskStatus.PENDING:
                return False
            task.status = TaskStatus.PROCESSING
            task.start_time = datetime.now()
            self._tasks.move_to_end(task_id)
            return True

    def complete_task(self, task_id: str, save_result_path: Optional[str] = None) -> None:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None or task.status in TERMINAL_STATES:
                return  # already cancelled/failed -> don't overwrite
            task.status = TaskStatus.COMPLETED
            task.end_time = datetime.now()
            if save_result_path:
                task.save_result_path = save_result_path
            self.completed_tasks += 1

    def fail_task(self, task_id: str, error: str, error_type: Optional[str] = None) -> None:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None or task.status in TERMINAL_STATES:
                return
            task.status = TaskStatus.FAILED
            task.end_time = datetime.now()
            task.error = error
            task.error_type = error_type
            self.failed_tasks += 1

    def cancel_task(self, task_id: str) -> bool:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None or task.status in TERMINAL_STATES:
                return False
            # A PENDING task is simply never started. A PROCESSING task sees
            # stop_event through the pipeline's cancelled() callback and raises
            # InterruptedError within one AR token / one flow-matching step.
            task.stop_event.set()
            task.status = TaskStatus.CANCELLED
            task.end_time = datetime.now()
            task.error = "Task cancelled by user"
            return True

    def cancel_all_tasks(self) -> None:
        with self._lock:
            for task_id, task in list(self._tasks.items()):
                if task.status in (TaskStatus.PENDING, TaskStatus.PROCESSING):
                    self.cancel_task(task_id)

    # -------------------------------------------------------------------- reads
    def get_task(self, task_id: str) -> Optional[TaskInfo]:
        with self._lock:
            return self._tasks.get(task_id)

    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        task = self.get_task(task_id)
        if not task:
            return None
        payload = {
            "task_id": task.task_id,
            "status": task.status.value,
            "start_time": task.start_time,
            "end_time": task.end_time,
            "error": task.error,
            "error_type": task.error_type or "",
            "save_result_path": task.save_result_path,
        }
        # Only emit progress keys once there is something to say.
        if task.phase is not None:
            payload["phase"] = task.phase
        if task.progress is not None:
            payload["progress"] = round(task.progress, 1)
        return payload

    def set_progress(
        self, task_id: str, phase: str, progress: Optional[float] = None
    ) -> None:
        """Record where a running task is: a phase label and, optionally, the
        global 0-99 progress. Called from the worker thread while the HTTP
        thread reads it, hence the lock.

        Never lets the global value move backwards (the facade forces
        monotonicity too, but a bar that dips between two polls of the engine
        itself would still read as a bug), and silently ignores
        unknown/terminal tasks: progress is best-effort telemetry and must never
        be able to fail a generation that is otherwise fine (e.g. a task
        cancelled between two token callbacks).
        """
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None or task.status in TERMINAL_STATES:
                return
            task.phase = phase
            if progress is not None:
                value = max(0.0, min(99.0, float(progress)))
                task.progress = value if task.progress is None else max(task.progress, value)

    def get_active_task_count(self) -> int:
        with self._lock:
            return sum(
                1
                for t in self._tasks.values()
                if t.status in (TaskStatus.PENDING, TaskStatus.PROCESSING)
            )

    def get_pending_task_count(self) -> int:
        with self._lock:
            return sum(1 for t in self._tasks.values() if t.status == TaskStatus.PENDING)

    def is_processing(self) -> bool:
        with self._lock:
            return self._current_processing_task is not None

    @property
    def current_task(self) -> Optional[str]:
        with self._lock:
            return self._current_processing_task

    # -------------------------------------------------------------- worker glue
    def acquire_processing_lock(self, task_id: str, timeout: Optional[float] = None) -> bool:
        acquired = self._processing_lock.acquire(timeout=timeout if timeout else -1)
        if acquired:
            with self._lock:
                self._current_processing_task = task_id
        return acquired

    def release_processing_lock(self, task_id: str) -> None:
        with self._lock:
            if self._current_processing_task == task_id:
                self._current_processing_task = None
                try:
                    self._processing_lock.release()
                except RuntimeError as e:
                    logger.warning(f"Task {task_id} lock release failed: {e}")

    def wait_for_next_pending_task(self, timeout: Optional[float] = None) -> Optional[str]:
        with self._task_available:
            task_id = self._get_next_pending_unlocked()
            if task_id:
                return task_id
            self._task_available.wait(timeout=timeout)
            return self._get_next_pending_unlocked()

    def _get_next_pending_unlocked(self) -> Optional[str]:
        for task_id, task in self._tasks.items():
            if task.status == TaskStatus.PENDING:
                return task_id
        return None

    def _cleanup_old_tasks(self, keep_count: int = 1000) -> None:
        if len(self._tasks) <= keep_count:
            return
        finished = [
            (tid, t) for tid, t in self._tasks.items() if t.status in TERMINAL_STATES
        ]
        finished.sort(key=lambda kv: kv[1].end_time or kv[1].start_time)
        for tid, _ in finished[: len(self._tasks) - keep_count]:
            del self._tasks[tid]
