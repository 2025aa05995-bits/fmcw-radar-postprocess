"""Angle spectrum measurement tab."""

from __future__ import annotations

from tkinter import ttk

import numpy as np

from ..frame import RadarFrame
from .base import MeasurementTab, register_tab
from .plotting import EmbeddedFigure, RxChannelSelector


@register_tab
class AngleSpectrumTab(MeasurementTab):
    """Live azimuth spectrum at the peak range-Doppler cell (selected RX)."""

    title = "Angle"
    order = 40

    def build(self, parent: ttk.Frame) -> None:
        opts = ttk.Frame(parent)
        opts.pack(side="top", fill="x", pady=(0, 4))
        ttk.Label(
            opts,
            text="Peak bin from selected RX RD map; unselected RX zeroed for angle FFT.",
        ).pack(side="left")
        self.rx_sel = RxChannelSelector(opts)

        self.plot = EmbeddedFigure(parent, figsize=(9.0, 4.2))
        self._line = None

    def update(self, frame: RadarFrame) -> None:
        if not self.is_visible:
            return
        self.rx_sel.sync(frame.cube.shape[2])
        rx_list = self.rx_sel.selected()

        rd_map = frame.rd_map_for_rx(rx_list)
        if not rx_list or not np.any(rd_map):
            ax = self.plot.axes
            ax.clear()
            self._line = None
            ax.set_title(f"Angle ({self.rx_sel.label()})  |  frame {frame.frame_id}")
            self.plot.draw_idle()
            return

        peak = np.unravel_index(int(np.argmax(rd_map)), rd_map.shape)
        range_idx, doppler_idx = int(peak[0]), int(peak[1])

        # Preserve array geometry: zero unselected RX before angle FFT.
        rd_sel = np.array(frame.rd_cube, copy=True)
        mask = np.ones(rd_sel.shape[2], dtype=bool)
        mask[rx_list] = False
        rd_sel[:, :, mask] = 0

        angle_cube = frame.processor.angle_fft(rd_sel)
        spectrum = np.abs(angle_cube[range_idx, doppler_idx, :])
        angle_axis = frame.processor.compute_angle_axis(angle_cube.shape[2])
        y = 20 * np.log10(spectrum + 1e-12)

        ax = self.plot.axes
        if self._line is None:
            ax.clear()
            (self._line,) = ax.plot(angle_axis, y, lw=1.4)
            ax.set_xlabel("Azimuth (deg)")
            ax.set_ylabel("Magnitude (dB)")
            ax.grid(True, alpha=0.3)
        else:
            self._line.set_data(angle_axis, y)
            ax.relim()
            ax.autoscale_view()

        r_m = frame.range_axis()[range_idx]
        v = frame.doppler_axis()[doppler_idx]
        ax.set_title(
            f"Angle @ R={r_m:.1f} m, v={v:.1f} m/s ({self.rx_sel.label()})  |  "
            f"frame {frame.frame_id}"
        )
        self.plot.draw_idle()
