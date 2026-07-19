"""Raw ADC time-domain measurement tab."""

from __future__ import annotations

from tkinter import ttk

import numpy as np

from ..frame import RadarFrame
from .base import MeasurementTab, register_tab
from .plotting import EmbeddedFigure, RxChannelSelector


@register_tab
class RawAdcTab(MeasurementTab):
    """Live fast-time ADC waveforms for selected RX channels."""

    title = "Raw ADC"
    order = 10

    def build(self, parent: ttk.Frame) -> None:
        opts = ttk.Frame(parent)
        opts.pack(side="top", fill="x", pady=(0, 4))
        ttk.Label(opts, text="Chirp:").pack(side="left")
        self.chirp_var = ttk.Spinbox(opts, from_=0, to=1023, width=6)
        self.chirp_var.set("0")
        self.chirp_var.pack(side="left", padx=4)
        self.rx_sel = RxChannelSelector(opts)

        self.plot = EmbeddedFigure(parent, figsize=(9.0, 4.2))
        self._lines: list = []
        self._last_rx: list[int] = []

    def update(self, frame: RadarFrame) -> None:
        if not self.is_visible:
            return
        self.rx_sel.sync(frame.cube.shape[2])
        rx_list = self.rx_sel.selected()
        try:
            chirp_idx = int(self.chirp_var.get())
        except ValueError:
            chirp_idx = 0
        chirp_idx = int(np.clip(chirp_idx, 0, frame.cube.shape[1] - 1))

        t_us = np.arange(frame.cube.shape[0]) / frame.config.adc_sample_rate_hz * 1e6
        ax = self.plot.axes
        rebuild = self._lines == [] or self._last_rx != rx_list
        if rebuild:
            ax.clear()
            self._lines = []
            for rx in rx_list:
                (ln,) = ax.plot(
                    t_us, frame.cube[:, chirp_idx, rx], lw=0.9, label=f"RX {rx}"
                )
                self._lines.append(ln)
            ax.set_xlabel("Time (µs)")
            ax.set_ylabel("ADC counts")
            ax.grid(True, alpha=0.3)
            if rx_list:
                ax.legend(loc="upper right", ncol=min(len(rx_list), 4), fontsize=8)
            self._last_rx = list(rx_list)
        else:
            for ln, rx in zip(self._lines, rx_list):
                ln.set_data(t_us, frame.cube[:, chirp_idx, rx])
            ax.relim()
            ax.autoscale_view()

        ax.set_title(
            f"Raw ADC — chirp {chirp_idx} ({self.rx_sel.label()})  |  frame {frame.frame_id}"
        )
        self.plot.draw_idle()
