"""Range-Doppler map measurement tab with max-hold."""

from __future__ import annotations

import tkinter as tk
from tkinter import BooleanVar, ttk

import numpy as np

from ..frame import RadarFrame
from ..theme import bordered_panel, panel_content, style_tk_scale
from .base import MeasurementTab, register_tab
from .plotting import EmbeddedFigure, RxChannelSelector

# Max unambiguous velocity slider (m/s) — maps to chirp period / idle time.
_VMAX_MPS_MIN = 5.0
_VMAX_MPS_MAX = 100.0
_VMAX_MPS_STEP = 0.5

# Velocity resolution slider (m/s) — maps to chirps-per-frame.
_DV_MPS_MIN = 0.05
_DV_MPS_MAX = 5.0
_DV_MPS_STEP = 0.05


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

        # Velocity controls under the plot (max velocity | velocity resolution).
        controls = ttk.Frame(parent)
        controls.pack(side="top", fill="x", pady=(6, 0))
        controls.columnconfigure(0, weight=1)
        controls.columnconfigure(1, weight=1)

        # --- Max velocity ---
        vmax_panel = bordered_panel(controls)
        vmax_panel.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        vmax_track = panel_content(vmax_panel)
        vmax_hdr = ttk.Frame(vmax_track)
        vmax_hdr.pack(side="top", fill="x")
        ttk.Label(vmax_hdr, text=f"{_VMAX_MPS_MIN:g}", font=("", 8)).pack(side="left")
        self._vmax_value_lbl = ttk.Label(
            vmax_hdr, text=f"Max velocity {_VMAX_MPS_MIN:.1f} m/s", font=("", 8)
        )
        self._vmax_value_lbl.pack(side="left", expand=True, padx=4)
        ttk.Label(vmax_hdr, text=f"{_VMAX_MPS_MAX:g} m/s", font=("", 8)).pack(side="right")

        self.vmax_var = tk.DoubleVar(value=float(_VMAX_MPS_MIN))
        self.vmax_scale = tk.Scale(
            vmax_track,
            from_=_VMAX_MPS_MIN,
            to=_VMAX_MPS_MAX,
            resolution=_VMAX_MPS_STEP,
            orient="horizontal",
            variable=self.vmax_var,
            showvalue=0,
            length=220,
            command=self._on_vmax_changed,
        )
        self.vmax_scale.pack(side="top", fill="x")
        style_tk_scale(self.vmax_scale)
        self._vmax_status = ttk.Label(vmax_track, text="", font=("", 8))
        self._vmax_status.pack(side="top")

        # --- Velocity resolution ---
        dv_panel = bordered_panel(controls)
        dv_panel.grid(row=0, column=1, sticky="ew", padx=(4, 0))
        dv_track = panel_content(dv_panel)
        dv_hdr = ttk.Frame(dv_track)
        dv_hdr.pack(side="top", fill="x")
        ttk.Label(dv_hdr, text=f"{_DV_MPS_MIN:g}", font=("", 8)).pack(side="left")
        self._dv_value_lbl = ttk.Label(
            dv_hdr, text=f"Δv {_DV_MPS_MIN:.2f} m/s", font=("", 8)
        )
        self._dv_value_lbl.pack(side="left", expand=True, padx=4)
        ttk.Label(dv_hdr, text=f"{_DV_MPS_MAX:g} m/s", font=("", 8)).pack(side="right")

        self.dv_var = tk.DoubleVar(value=float(_DV_MPS_MIN))
        self.dv_scale = tk.Scale(
            dv_track,
            from_=_DV_MPS_MIN,
            to=_DV_MPS_MAX,
            resolution=_DV_MPS_STEP,
            orient="horizontal",
            variable=self.dv_var,
            showvalue=0,
            length=220,
            command=self._on_dv_changed,
        )
        self.dv_scale.pack(side="top", fill="x")
        style_tk_scale(self.dv_scale)
        self._dv_status = ttk.Label(dv_track, text="", font=("", 8))
        self._dv_status.pack(side="top")

        self._image = None
        self._cbar = None
        self._hold_map: np.ndarray | None = None
        self._last_rx: list[int] = []
        self._last_vmax_mps: float | None = None
        self._last_dv_mps: float | None = None
        self._syncing_velocity = False
        parent.after_idle(self._sync_velocity_from_device)

    def reset_max_hold(self) -> None:
        """Clear accumulated max-hold map."""
        self._hold_map = None

    def _on_max_hold_toggle(self) -> None:
        if self.max_hold.get():
            self.reset_max_hold()

    def _snap_vmax(self, v: float) -> float:
        stepped = round(float(v) / _VMAX_MPS_STEP) * _VMAX_MPS_STEP
        return float(np.clip(stepped, _VMAX_MPS_MIN, _VMAX_MPS_MAX))

    def _snap_dv(self, dv: float) -> float:
        stepped = round(float(dv) / _DV_MPS_STEP) * _DV_MPS_STEP
        return float(np.clip(stepped, _DV_MPS_MIN, _DV_MPS_MAX))

    def _vmax_mps(self) -> float:
        try:
            return self._snap_vmax(float(self.vmax_var.get()))
        except (TypeError, ValueError, tk.TclError):
            return _VMAX_MPS_MIN

    def _dv_mps(self) -> float:
        try:
            return self._snap_dv(float(self.dv_var.get()))
        except (TypeError, ValueError, tk.TclError):
            return _DV_MPS_MIN

    def _sync_velocity_from_device(self) -> None:
        """Initialize velocity sliders from the live device config."""
        device = self.device
        if device is None:
            return
        cfg = device.config
        vmax = self._snap_vmax(float(cfg.max_velocity_mps))
        dv = self._snap_dv(float(cfg.doppler_resolution_mps))
        self._syncing_velocity = True
        try:
            self.vmax_var.set(vmax)
            self.dv_var.set(dv)
        finally:
            self._syncing_velocity = False
        self._last_vmax_mps = None
        self._last_dv_mps = None
        self._on_vmax_changed()
        self._on_dv_changed()

    def _on_vmax_changed(self, _value=None) -> None:
        """Push max velocity → ``device.updateMaxVelocity`` (chirp period)."""
        if self._syncing_velocity:
            return
        vmax = self._vmax_mps()
        self._vmax_value_lbl.configure(text=f"Max velocity {vmax:.1f} m/s")
        if self._last_vmax_mps is not None and abs(self._last_vmax_mps - vmax) < 1e-9:
            return
        self._last_vmax_mps = vmax
        device = self.device
        if device is None:
            self._vmax_status.configure(text="(no device)")
            return
        try:
            device.updateMaxVelocity(vmax)
            # Keep Δv target: re-apply resolution after T_c changes.
            device.updateVelocityResolution(self._dv_mps())
            self._last_dv_mps = self._dv_mps()
        except Exception as exc:  # noqa: BLE001
            self._vmax_status.configure(text=f"error: {exc}")
            return
        n = int(device.config.num_chirps)
        t_c_us = device.config.chirp_period_s * 1e6
        self._vmax_status.configure(text=f"T_c={t_c_us:.1f} µs, N={n}")
        self._dv_status.configure(text=f"N={n} chirps")
        self.reset_max_hold()

    def _on_dv_changed(self, _value=None) -> None:
        """Push velocity resolution → ``device.updateVelocityResolution``."""
        if self._syncing_velocity:
            return
        dv = self._dv_mps()
        self._dv_value_lbl.configure(text=f"Δv {dv:.2f} m/s")
        if self._last_dv_mps is not None and abs(self._last_dv_mps - dv) < 1e-9:
            return
        self._last_dv_mps = dv
        device = self.device
        if device is None:
            self._dv_status.configure(text="(no device)")
            return
        try:
            device.updateVelocityResolution(dv)
        except Exception as exc:  # noqa: BLE001
            self._dv_status.configure(text=f"error: {exc}")
            return
        n = int(device.config.num_chirps)
        self._dv_status.configure(text=f"N={n} chirps → device")
        self.reset_max_hold()

    def _teardown_heatmap(self) -> None:
        """Remove image + colorbar cleanly (avoids stacked colorbar axes)."""
        if self._cbar is not None:
            try:
                self._cbar.remove()
            except Exception:  # noqa: BLE001
                pass
            self._cbar = None
        self._image = None
        ax = self.plot.axes
        ax.clear()
        # Colorbar creates a sibling axes; purge any that are not the main plot.
        for extra in list(self.plot.fig.axes):
            if extra is not ax:
                try:
                    self.plot.fig.delaxes(extra)
                except Exception:  # noqa: BLE001
                    pass

    def on_device_changed(self) -> None:
        """Re-sync velocity controls from the newly selected device."""
        self._last_vmax_mps = None
        self._last_dv_mps = None
        self._sync_velocity_from_device()

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
        clim = (float(rd_db.min()), float(rd_db.max()) + 1e-12)

        ax = self.plot.axes
        if self._image is None:
            self._teardown_heatmap()
            self._image = ax.imshow(rd_db, aspect="auto", extent=extent, cmap="viridis")
            self._cbar = self.plot.fig.colorbar(self._image, ax=ax, label="dB")
            ax.set_xlabel("Velocity (m/s)")
            ax.set_ylabel("Range (m)")
        else:
            # In-place update — set_data accepts a new shape when N / FFT size changes.
            self._image.set_data(rd_db)
            self._image.set_extent(extent)
            self._image.set_clim(*clim)

        mode = "max hold" if self.max_hold.get() else "live"
        vmax = frame.config.max_velocity_mps
        dv = frame.config.doppler_resolution_mps
        ax.set_title(
            f"Range-Doppler [{mode}]  vmax={vmax:.1f}  Δv={dv:.2f} m/s  "
            f"({self.rx_sel.label()})  |  frame {frame.frame_id}"
        )
        self.plot.draw_idle()
