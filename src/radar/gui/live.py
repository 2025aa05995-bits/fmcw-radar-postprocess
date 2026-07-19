"""Continuous live data worker backed by a ``RadarDevice``."""

from __future__ import annotations

import threading
import time
from queue import Empty, Full, Queue

from ..devices import RadarDevice
from ..testing import QAThresholds
from .frame import RadarFrame


class LiveDataWorker:
    """
    Background thread that continuously reads cubes from a ``RadarDevice``.

    Queue size is 1 so stale frames are dropped and the UI always shows the
    newest data. The device is opened on ``start()`` and closed on ``stop()``.

    Construct devices via ``RadarDeviceFactory.create(...)`` — config and
    driver options stay in the factory/device layer.

    RF control changes (HPF / LPF / power / gain) nudge the worker so a fresh
    frame is captured immediately instead of waiting for the next interval.
    """

    def __init__(
        self,
        device: RadarDevice,
        *,
        interval_s: float = 0.1,
        qa_thresholds: QAThresholds | None = None,
    ) -> None:
        self.device = device
        self.interval_s = max(0.02, float(interval_s))
        self.qa_thresholds = qa_thresholds or QAThresholds()
        self._queue: Queue[RadarFrame] = Queue(maxsize=1)
        self._stop = threading.Event()
        self._nudge = threading.Event()
        self._thread: threading.Thread | None = None
        self._frame_id = 0
        self._error: str | None = None
        self._lock = threading.Lock()
        self._bind_device_callbacks(device)

    @property
    def last_error(self) -> str | None:
        with self._lock:
            return self._error

    def _bind_device_callbacks(self, device: RadarDevice) -> None:
        """Wire RF-control updates to an immediate frame nudge."""
        device.set_controls_changed_callback(self.nudge)

    def nudge(self) -> None:
        """Request an immediate frame capture (e.g. after RF control change)."""
        self._nudge.set()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        if not self.device.is_open:
            self.device.open()
        self._bind_device_callbacks(self.device)
        self._stop.clear()
        self._nudge.clear()
        self._thread = threading.Thread(target=self._run, name="LiveDataWorker", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        self._nudge.set()  # wake sleep waiters
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None
        if self.device.is_open:
            try:
                self.device.close()
            except Exception as exc:  # noqa: BLE001
                with self._lock:
                    self._error = f"close failed: {exc}"

    def set_interval(self, interval_s: float) -> None:
        self.interval_s = max(0.02, float(interval_s))

    def set_device(self, device: RadarDevice) -> None:
        """
        Hot-swap the device.

        The worker must be stopped (or will stop the old device) before the
        next ``start()`` opens the new one.
        """
        was_running = self._thread is not None and self._thread.is_alive()
        if was_running:
            self.stop()
        try:
            self.device.set_controls_changed_callback(None)
        except Exception:  # noqa: BLE001
            pass
        self.device = device
        self._bind_device_callbacks(device)
        self._frame_id = 0
        if was_running:
            self.start()

    def get_latest(self, timeout: float = 0.0) -> RadarFrame | None:
        try:
            if timeout > 0:
                return self._queue.get(timeout=timeout)
            return self._queue.get_nowait()
        except Empty:
            return None

    def _run(self) -> None:
        while not self._stop.is_set():
            t0 = time.perf_counter()
            try:
                if not self.device.is_open:
                    self.device.open()
                cube = self.device.read_frame(self._frame_id)
                temperature_c = None
                if self.device.supports_temperature():
                    try:
                        temperature_c = self.device.read_temperature()
                    except Exception:  # noqa: BLE001 — temp is optional telemetry
                        temperature_c = None
                frame = RadarFrame(
                    cube=cube,
                    config=self.device.config,
                    frame_id=self._frame_id,
                    source_name=self.device.name,
                    temperature_c=temperature_c,
                    qa_thresholds=self.qa_thresholds,
                )
                self._frame_id += 1
                with self._lock:
                    self._error = None
                self._put_latest(frame)
            except Exception as exc:  # noqa: BLE001 — surface to GUI status bar
                with self._lock:
                    self._error = str(exc)
            elapsed = time.perf_counter() - t0
            wait_s = max(0.0, self.interval_s - elapsed)
            self._nudge.clear()
            # Wake early when RF controls change so synthetic data reacts ASAP.
            self._nudge.wait(timeout=wait_s)

    def _put_latest(self, frame: RadarFrame) -> None:
        """Keep only the newest frame; drop anything the UI hasn't consumed."""
        try:
            self._queue.put_nowait(frame)
        except Full:
            try:
                self._queue.get_nowait()
            except Empty:
                pass
            try:
                self._queue.put_nowait(frame)
            except Full:
                pass
