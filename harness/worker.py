"""Background task worker for the YuE2 async task server.

A single daemon thread pulls PENDING tasks FIFO and runs the (blocking,
non-reentrant) generation serially — matching LightX2V's single processing
thread. Cancellation is real, not best-effort: the task's stop_event is handed
to the pipeline as its ``cancelled()`` callback, so a DELETE lands within one
AR token / one flow-matching step and the pipeline raises InterruptedError.
"""
import logging
import os
import threading
from typing import Callable

from .schemas import MusicTaskRequest
from .task_manager import TaskManager, TaskStatus

logger = logging.getLogger(__name__)

# generate(request, cancelled) -> None ; raises on failure (InterruptedError on
# cancel); writes request.save_result_path atomically.
GenerateFn = Callable[[MusicTaskRequest, Callable[[], bool]], None]


class TaskWorker:
    def __init__(self, task_manager: TaskManager, generate: GenerateFn):
        self._tm = task_manager
        self._generate = generate
        self._stop = threading.Event()
        self._thread: threading.Thread = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="yue2-task-worker")
        self._thread.start()
        logger.info("YuE2 task worker started")

    def stop(self) -> None:
        # Deliberately daemon + short join: the engine runs in a GPUStack-managed
        # container whose runtime SIGKILLs after the stop grace period, so no
        # thread design can outlive a long generation anyway. An abrupt kill is a
        # first-class recoverable event upstream: the facade's death-requeue
        # sweeper re-dispatches the in-flight task (fresh save_result_path) and
        # the storage janitor TTL-reaps any stale partial output.
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)

    def _loop(self) -> None:
        while not self._stop.is_set():
            task_id = self._tm.wait_for_next_pending_task(timeout=1.0)
            if not task_id:
                continue
            task = self._tm.get_task(task_id)
            if not task or task.status != TaskStatus.PENDING:
                continue
            self._process(task_id)

    def _process(self, task_id: str) -> None:
        if not self._tm.acquire_processing_lock(task_id, timeout=1):
            self._tm.fail_task(task_id, "failed to acquire processing lock")
            return
        try:
            task = self._tm.get_task(task_id)
            if task is None:
                return
            # Atomic PENDING -> PROCESSING; False means cancelled (or otherwise
            # already handled) between the pending check and here — skip without
            # touching its (terminal) state.
            if not self._tm.start_task(task_id):
                logger.info(f"Task {task_id} not started (cancelled before processing)")
                return

            try:
                self._generate(task.message, task.stop_event.is_set)
            except InterruptedError:
                # Only a DELETE sets stop_event, and cancel_task has already
                # moved the task to CANCELLED — nothing to record here. Any other
                # InterruptedError is a real failure and must not look cancelled.
                if not task.stop_event.is_set():
                    raise
                logger.info(f"Task {task_id} cancelled during processing; generation stopped")
                return

            # Cancelled after the last cancelled() poll but before we got here
            # (i.e. during encode/save): leave it CANCELLED and best-effort drop
            # the already-written output so no orphan result lingers at the final
            # path (janitor TTL remains the backstop).
            cur = self._tm.get_task(task_id)
            if cur is not None and cur.status == TaskStatus.CANCELLED:
                logger.info(f"Task {task_id} cancelled after generation; result discarded")
                if task.save_result_path:
                    try:
                        os.remove(task.save_result_path)
                    except OSError:
                        pass
                return

            self._tm.complete_task(task_id, save_result_path=task.save_result_path)
            logger.info(f"Task {task_id} completed -> {task.save_result_path}")
        except Exception as e:  # noqa: BLE001
            logger.exception(f"Task {task_id} failed: {e}")
            self._tm.fail_task(task_id, str(e), error_type=getattr(e, "error_type", type(e).__name__))
        finally:
            self._tm.release_processing_lock(task_id)
