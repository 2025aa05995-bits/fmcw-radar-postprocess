"""Range-Doppler map measurement tab with max-hold."""

from __future__ import annotations

from tkinter import BooleanVar, ttk

import numpy as np

from ..frame import RadarFrame
from .base import MeasurementTab, register_tab
from .plotting import EmbeddedFigure, RxChannelSelector


@register_tab
class RangeDopplerTab(MeasurementTab):
    """Live range-Doppler heatmap from selected RX channels."""

    title = "Range-Doppler"
    order = 30

    def build(self, parent: ttk.Frame) -> None:
        opts = ttk.Frame(parent)
        opts.pack(side="top", fill="x", pady=(0, 4))

        self.max_hold = BooleanVar(value=False)
        ttk.Checkbutton(
            opts, text="Max hold", variable=self.max_hold, command=self._on_max_hold_toggle
        ).pack(side="left", padx=(0, 4))
        ttk.Button(opts, text="Reset hold", command=self.reset_max_hold).pack(
            side="left", padx=2
        )

        self.rx_sel = RxChannelSelector(opts)
        self.plot = EmbeddedFigure(parent, figsize=(9.0, 5.0))
        self._image = None
        self._cbar = None
        self._hold_map: np.ndarray | None = None
        self._last_rx: list[int] = []

    def reset_max_hold(self) -> None:
        """Clear accumulated max-hold map."""
        self._hold_map = None

    def _on_max_hold_toggle(self) -> None:
        if self.max_hold.get():
            self.reset_max_hold()

    def update(self, frame: RadarFrame) -> None:
        if not self.is_visible:
            return
        self.rx_sel.sync(frame.cube.shape[2])
        rx_list = self.rx_sel.selected()
        if rx_list != self._last_rx:
            self.reset_max_hold()
            self._last_rx = list(rx_list)

        rd_map = frame.rd_map_for_rx(rx_list)

        if self.max_hold.get():
            if self._hold_map is None or self._hold_map.shape != rd_map.shape:
                self._hold_map = rd_map.astype(np.float64, copy=True)
            else:
                np.maximum(self._hold_map, rd_map, out=self._hold_map)
            display = self._hold_map
        else:
            display = rd_map

        rd_db = 20 * np.log10(display + 1e-12)
        r_axis = frame.range_axis()
        d_axis = frame.doppler_axis()
        extent = [d_axis[0], d_axis[-1], r_axis[-1], r_axis[0]]

        ax = self.plot.axes
        if self._image is None:
            ax.clear()
            self._image = ax.imshow(rd_db, aspect="auto", extent=extent, cmap="viridis")
            self._cbar = self.plot.fig.colorbar(self._image, ax=ax, label="dB")
            ax.set_xlabel("Velocity (m/s)")
            ax.set_ylabel("Range (m)")
        else:
            self._image.set_data(rd_db)
            self._image.set_extent(extent)
            self._image.set_clim(float(rd_db.min()), float(rd_db.max()) + 1e-12)

        mode = "max hold" if self.max_hold.get() else "live"
        ax.set_title(
            f"Range-Doppler [{mode}] ({self.rx_sel.label()})  |  frame {frame.frame_id}"
        )
        self.plot.draw_idle()
