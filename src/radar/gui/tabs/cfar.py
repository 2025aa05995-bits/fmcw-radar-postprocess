"""CFAR detections overlay on the live range-Doppler map."""

from __future__ import annotations

from tkinter import BooleanVar, ttk

import numpy as np

from ...process import RadarProcessor
from ..frame import RadarFrame
from .base import MeasurementTab, register_tab
from .plotting import EmbeddedFigure, RxChannelSelector

# Defaults match ``RadarProcessor.cfar_2d``.
_DEFAULT_GUARD_R = 2
_DEFAULT_GUARD_D = 2
_DEFAULT_TRAIN_R = 8
_DEFAULT_TRAIN_D = 8
_DEFAULT_PFA = 1e-4
_DEFAULT_THRESHOLD_DB = 12.0


@register_tab
class CfarTab(MeasurementTab):
    """Live 2D CFAR detections on the range-Doppler map (selected RX)."""

    title = "CFAR"
    order = 35

    def build(self, parent: ttk.Frame) -> None:
        opts = ttk.Frame(parent)
        opts.pack(side="top", fill="x", pady=(0, 4))

        # --- CA-CFAR window geometry ---
        ttk.Label(opts, text="Guard R:").pack(side="left")
        self.guard_r = ttk.Spinbox(opts, from_=0, to=64, width=4)
        self.guard_r.set(str(_DEFAULT_GUARD_R))
        self.guard_r.pack(side="left", padx=(2, 6))

        ttk.Label(opts, text="Guard D:").pack(side="left")
        self.guard_d = ttk.Spinbox(opts, from_=0, to=64, width=4)
        self.guard_d.set(str(_DEFAULT_GUARD_D))
        self.guard_d.pack(side="left", padx=(2, 6))

        ttk.Label(opts, text="Train R:").pack(side="left")
        self.train_r = ttk.Spinbox(opts, from_=1, to=128, width=4)
        self.train_r.set(str(_DEFAULT_TRAIN_R))
        self.train_r.pack(side="left", padx=(2, 6))

        ttk.Label(opts, text="Train D:").pack(side="left")
        self.train_d = ttk.Spinbox(opts, from_=1, to=128, width=4)
        self.train_d.set(str(_DEFAULT_TRAIN_D))
        self.train_d.pack(side="left", padx=(2, 8))

        ttk.Separator(opts, orient="vertical").pack(side="left", fill="y", padx=6)

        # --- Threshold mode: Pfa (adaptive) or fixed dB above noise ---
        ttk.Label(opts, text="Pfa:").pack(side="left")
        self.pfa_var = ttk.Entry(opts, width=9)
        self.pfa_var.insert(0, f"{_DEFAULT_PFA:g}")
        self.pfa_var.pack(side="left", padx=(2, 6))

        self.use_fixed_thresh = BooleanVar(value=False)
        ttk.Checkbutton(
            opts,
            text="Fixed thresh",
            variable=self.use_fixed_thresh,
            command=self._on_thresh_mode_toggle,
        ).pack(side="left", padx=(4, 2))
        ttk.Label(opts, text="dB:").pack(side="left")
        self.threshold_db_var = ttk.Entry(opts, width=6)
        self.threshold_db_var.insert(0, f"{_DEFAULT_THRESHOLD_DB:g}")
        self.threshold_db_var.pack(side="left", padx=(2, 6))
        self._on_thresh_mode_toggle()

        ttk.Separator(opts, orient="vertical").pack(side="left", fill="y", padx=6)
        self.rx_sel = RxChannelSelector(opts)

        row2 = ttk.Frame(parent)
        row2.pack(side="top", fill="x", pady=(0, 4))
        self.show_threshold = BooleanVar(value=False)
        ttk.Checkbutton(
            row2,
            text="Show threshold map",
            variable=self.show_threshold,
        ).pack(side="left")
        ttk.Button(row2, text="Reset defaults", command=self._reset_defaults).pack(
            side="left", padx=8
        )
        self._settings_lbl = ttk.Label(row2, text="", style="Muted.TLabel")
        self._settings_lbl.pack(side="left", padx=8)

        self.plot = EmbeddedFigure(parent, figsize=(9.0, 5.0))
        self._image = None
        self._scatter = None
        self._cbar = None
        self._showing_threshold = False

    def _on_thresh_mode_toggle(self) -> None:
        """Enable fixed-dB entry only when that mode is selected."""
        state = "normal" if self.use_fixed_thresh.get() else "disabled"
        self.threshold_db_var.configure(state=state)
        pfa_state = "disabled" if self.use_fixed_thresh.get() else "normal"
        self.pfa_var.configure(state=pfa_state)

    def _teardown_heatmap(self) -> None:
        """Remove image + colorbar cleanly (avoids stacked colorbar axes)."""
        if self._cbar is not None:
            try:
                self._cbar.remove()
            except Exception:  # noqa: BLE001
                pass
            self._cbar = None
        self._image = None
        self._scatter = None
        ax = self.plot.axes
        ax.clear()
        for extra in list(self.plot.fig.axes):
            if extra is not ax:
                try:
                    self.plot.fig.delaxes(extra)
                except Exception:  # noqa: BLE001
                    pass
        self._showing_threshold = False

    def _reset_defaults(self) -> None:
        """Restore CFAR parameters to ``cfar_2d`` defaults."""
        self.guard_r.set(str(_DEFAULT_GUARD_R))
        self.guard_d.set(str(_DEFAULT_GUARD_D))
        self.train_r.set(str(_DEFAULT_TRAIN_R))
        self.train_d.set(str(_DEFAULT_TRAIN_D))
        self.pfa_var.configure(state="normal")
        self.pfa_var.delete(0, "end")
        self.pfa_var.insert(0, f"{_DEFAULT_PFA:g}")
        self.threshold_db_var.configure(state="normal")
        self.threshold_db_var.delete(0, "end")
        self.threshold_db_var.insert(0, f"{_DEFAULT_THRESHOLD_DB:g}")
        self.use_fixed_thresh.set(False)
        self.show_threshold.set(False)
        self._on_thresh_mode_toggle()

    def _read_int(self, widget: ttk.Spinbox, default: int, *, minimum: int = 0) -> int:
        try:
            value = int(float(widget.get()))
        except (TypeError, ValueError):
            return default
        return max(minimum, value)

    def _cfar_kwargs(self) -> dict:
        """Collect current UI settings into ``cfar_2d`` keyword arguments."""
        guard_r = self._read_int(self.guard_r, _DEFAULT_GUARD_R, minimum=0)
        guard_d = self._read_int(self.guard_d, _DEFAULT_GUARD_D, minimum=0)
        train_r = self._read_int(self.train_r, _DEFAULT_TRAIN_R, minimum=1)
        train_d = self._read_int(self.train_d, _DEFAULT_TRAIN_D, minimum=1)

        kwargs: dict = {
            "guard_cells": (guard_r, guard_d),
            "train_cells": (train_r, train_d),
        }
        if self.use_fixed_thresh.get():
            try:
                thresh_db = float(self.threshold_db_var.get())
            except ValueError:
                thresh_db = _DEFAULT_THRESHOLD_DB
            kwargs["threshold_db"] = thresh_db
            kwargs["pfa"] = _DEFAULT_PFA  # unused when threshold_db is set
        else:
            try:
                pfa = float(self.pfa_var.get())
            except ValueError:
                pfa = _DEFAULT_PFA
            kwargs["pfa"] = max(pfa, 1e-12)
            kwargs["threshold_db"] = None
        return kwargs

    def _settings_summary(self, kwargs: dict, n_det: int) -> str:
        gr, gd = kwargs["guard_cells"]
        tr, td = kwargs["train_cells"]
        if kwargs.get("threshold_db") is not None:
            mode = f"fixed {kwargs['threshold_db']:.1f} dB"
        else:
            mode = f"Pfa={kwargs['pfa']:g}"
        return f"G({gr},{gd})  T({tr},{td})  {mode}  ·  dets={n_det}"

    def update(self, frame: RadarFrame) -> None:
        if not self.is_visible:
            return
        self.rx_sel.sync(frame.cube.shape[2])

        kwargs = self._cfar_kwargs()
        rd_map = frame.rd_map_for_rx(self.rx_sel.selected())
        if rd_map.size == 0:
            return

        mask, threshold = RadarProcessor.cfar_2d(rd_map, **kwargs)
        show_thr = self.show_threshold.get()
        if show_thr:
            display = 20 * np.log10(np.maximum(threshold, 1e-12))
            # Non-evaluated border cells are +inf — hide them.
            display = np.where(np.isfinite(threshold), display, np.nan)
            cbar_label = "Threshold (dB)"
            cmap_name = "magma"
        else:
            display = 20 * np.log10(rd_map + 1e-12)
            cbar_label = "Magnitude (dB)"
            cmap_name = "viridis"

        r_axis = frame.range_axis()
        d_axis = frame.doppler_axis()
        extent = [d_axis[0], d_axis[-1], r_axis[-1], r_axis[0]]
        ri, di = np.where(mask)

        finite = display[np.isfinite(display)]
        if finite.size:
            clim = (float(np.nanmin(finite)), float(np.nanmax(finite)) + 1e-12)
        else:
            clim = (0.0, 1.0)

        ax = self.plot.axes
        if self._image is None:
            self._teardown_heatmap()
            self._image = ax.imshow(
                display,
                aspect="auto",
                extent=extent,
                cmap=cmap_name,
            )
            self._cbar = self.plot.fig.colorbar(self._image, ax=ax, label=cbar_label)
            self._scatter = ax.scatter(
                d_axis[di],
                r_axis[ri],
                c="#ff6b6b",
                s=18,
                marker="x",
                label="CFAR",
            )
            ax.set_xlabel("Velocity (m/s)")
            ax.set_ylabel("Range (m)")
            ax.legend(loc="upper right")
            self._showing_threshold = show_thr
        else:
            # In-place update — survives shape changes and threshold-map toggles
            # without stacking colorbar axes.
            self._image.set_data(display)
            self._image.set_extent(extent)
            self._image.set_clim(*clim)
            if self._showing_threshold != show_thr:
                self._image.set_cmap(cmap_name)
                self._showing_threshold = show_thr
            if self._cbar is not None:
                self._cbar.set_label(cbar_label)
            if self._scatter is not None:
                self._scatter.set_offsets(np.column_stack([d_axis[di], r_axis[ri]]))

        summary = self._settings_summary(kwargs, int(ri.size))
        self._settings_lbl.configure(text=summary)
        ax.set_title(
            f"CFAR ({self.rx_sel.label()})  |  {summary}  |  frame {frame.frame_id}"
        )
        self.plot.draw_idle()
