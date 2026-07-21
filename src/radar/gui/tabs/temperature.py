"""Device temperature history measurement tab."""

from __future__ import annotations

from collections import deque
from tkinter import BooleanVar, IntVar, ttk

import numpy as np

from ..frame import RadarFrame
from .base import MeasurementTab, register_tab
from .plotting import EmbeddedFigure


@register_tab
class TemperatureTab(MeasurementTab):
    """
    Live device temperature vs frame.

    When **Enable temperature plotting** is on, samples are collected on every
    live frame (even if another tab is active). The plot refreshes only while
    this tab is visible.
    """

    title = "Temperature"
    order = 55
    needs_background_ingest = True

    def build(self, parent: ttk.Frame) -> None:
        opts = ttk.Frame(parent)
        opts.pack(side="top", fill="x", pady=(0, 4))

        self.enable_plot = BooleanVar(value=False)
        ttk.Checkbutton(
            opts,
            text="Enable temperature plotting",
            variable=self.enable_plot,
            command=self._on_enable_toggle,
        ).pack(side="left", padx=(0, 8))

        ttk.Label(opts, text="History:").pack(side="left")
        self.history_len = IntVar(value=300)
        hist = ttk.Spinbox(
            opts, from_=50, to=5000, width=6, textvariable=self.history_len
        )
        hist.pack(side="left", padx=4)
        ttk.Label(opts, text="samples").pack(side="left")

        ttk.Button(opts, text="Clear", command=self.clear_history).pack(
            side="left", padx=8
        )

        self.status = ttk.Label(opts, text="Plotting disabled")
        self.status.pack(side="right", padx=8)

        self.plot = EmbeddedFigure(parent, figsize=(9.0, 4.0))
        self._line = None
        self._frames: deque[int] = deque()
        self._temps: deque[float] = deque()
        self._last_temp: float | None = None
        self._last_source: str = ""
        self._last_frame_id: int = -1
        self._unavailable = False

    def clear_history(self) -> None:
        self._frames.clear()
        self._temps.clear()
        self._last_temp = None
        self._last_frame_id = -1
        self._unavailable = False
        if self._line is not None:
            self._line.set_data([], [])
        ax = self.plot.axes
        ax.relim()
        ax.autoscale_view()
        if self.is_visible:
            self.plot.draw_idle()
        self.status.configure(text="History cleared")

    def _on_enable_toggle(self) -> None:
        if self.enable_plot.get():
            self.status.configure(text="Collecting — waiting for samples…")
        else:
            self.status.configure(text="Plotting disabled")

    def _maxlen(self) -> int:
        try:
            return max(10, int(self.history_len.get()))
        except (TypeError, ValueError):
            return 300

    def ingest_frame(self, frame: RadarFrame) -> None:
        """Collect temperature continuously while plotting is enabled."""
        if not self.enable_plot.get():
            return

        # Avoid double-append if the same frame is ingested twice.
        if frame.frame_id == self._last_frame_id:
            return

        self._last_source = frame.source_name
        if frame.temperature_c is None:
            self._unavailable = True
            return

        self._unavailable = False
        self._last_frame_id = frame.frame_id
        self._last_temp = float(frame.temperature_c)

        maxlen = self._maxlen()
        self._frames.append(frame.frame_id)
        self._temps.append(self._last_temp)
        while len(self._frames) > maxlen:
            self._frames.popleft()
            self._temps.popleft()

        if not self.is_visible:
            self.status.configure(
                text=f"Collecting {self._last_temp:.2f} °C  |  n={len(self._temps)}"
            )

    def on_show(self) -> None:
        super().on_show()
        # Redraw accumulated history when switching back to this tab.
        self._redraw()

    def update(self, frame: RadarFrame) -> None:
        """Ingest (if needed) and redraw only while this tab is active."""
        self.ingest_frame(frame)
        if not self.is_visible:
            return
        self._redraw()

    def _redraw(self) -> None:
        ax = self.plot.axes

        if not self.enable_plot.get():
            ax.clear()
            self._line = None
            ax.set_xlabel("Frame")
            ax.set_ylabel("Temperature (°C)")
            ax.set_title("Temperature (disabled)")
            ax.grid(True, alpha=0.3)
            self.plot.draw_idle()
            return

        if self._unavailable and not self._temps:
            ax.clear()
            self._line = None
            ax.set_title(
                f"Temperature unavailable ({self._last_source or 'device'})"
            )
            ax.set_xlabel("Frame")
            ax.set_ylabel("Temperature (°C)")
            ax.grid(True, alpha=0.3)
            self.status.configure(
                text=f"No temperature from device '{self._last_source}'"
            )
            self.plot.draw_idle()
            return

        if not self._temps:
            ax.clear()
            self._line = None
            ax.set_xlabel("Frame")
            ax.set_ylabel("Temperature (°C)")
            ax.set_title("Temperature — collecting…")
            ax.grid(True, alpha=0.3)
            self.plot.draw_idle()
            return

        x = np.fromiter(self._frames, dtype=np.int64, count=len(self._frames))
        y = np.fromiter(self._temps, dtype=np.float64, count=len(self._temps))

        if self._line is None:
            ax.clear()
            (self._line,) = ax.plot(x, y, lw=1.4, color="#e76f51", marker="o", ms=2)
            ax.set_xlabel("Frame")
            ax.set_ylabel("Temperature (°C)")
            ax.grid(True, alpha=0.3)
        else:
            self._line.set_data(x, y)
            ax.relim()
            ax.autoscale_view()

        t = self._last_temp if self._last_temp is not None else y[-1]
        ax.set_title(
            f"Temperature  {t:.2f} °C  ({self._last_source})  |  "
            f"frame {self._last_frame_id}"
        )
        self.status.configure(text=f"{t:.2f} °C  |  n={len(self._temps)}")
        self.plot.draw_idle()
