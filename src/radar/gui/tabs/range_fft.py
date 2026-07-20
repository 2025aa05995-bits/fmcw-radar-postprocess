"""Range FFT measurement tab with chirp / avg / max-hold modes."""

from __future__ import annotations

import tkinter as tk
from tkinter import BooleanVar, StringVar, ttk

import numpy as np
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk

from ...devices import SPEED_OF_LIGHT
from ..frame import RadarFrame
from ..theme import COLORS, bordered_panel, panel_content, style_mpl_toolbar, style_tk_scale
from .base import MeasurementTab, register_tab
from .plotting import EmbeddedFigure, RxChannelSelector

# HPF slider span (kHz).
_HPF_KHZ_MIN = 0
_HPF_KHZ_MAX = 3600
_HPF_KHZ_STEP = 200

# LPF slider span (MHz) — compact control on the right under the plot.
_LPF_MHZ_MIN = 15
_LPF_MHZ_MAX = 40
_LPF_MHZ_STEP = 1

# TX power / RX gain (dB) — vertical side sliders.
_TX_PWR_DB_MIN = 0
_TX_PWR_DB_MAX = 15
_TX_PWR_DB_STEP = 1
_RX_GAIN_DB_VALUES = tuple(range(26, 49, 3))  # 26, 29, … 47 (≤ 48, 3 dB)
_RX_GAIN_DB_MIN = _RX_GAIN_DB_VALUES[0]
_RX_GAIN_DB_MAX = _RX_GAIN_DB_VALUES[-1]
_RX_GAIN_DB_STEP = 3

# Range-resolution slider ↔ chirp bandwidth (δR = c / 2B).
_CHIRP_BW_HZ_MIN = 200e6
_CHIRP_BW_HZ_MAX = 4e9
_DR_M_FINE = SPEED_OF_LIGHT / (2.0 * _CHIRP_BW_HZ_MAX)  # ~0.0375 m
_DR_M_COARSE = SPEED_OF_LIGHT / (2.0 * _CHIRP_BW_HZ_MIN)  # ~0.75 m
_DR_M_STEP = 0.01


@register_tab
class RangeFftTab(MeasurementTab):
    """Live range FFT profiles for selected RX (chirp / avg / holds)."""

    title = "Range FFT"
    order = 20

    def build(self, parent: ttk.Frame) -> None:
        # Grid: row 1 (plot) expands; opts / footer stay natural height.
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)

        opts = ttk.Frame(parent)
        opts.grid(row=0, column=0, sticky="ew", pady=(0, 4))

        ttk.Label(opts, text="Chirp:").pack(side="left")
        self.chirp_var = ttk.Spinbox(opts, from_=0, to=1023, width=6)
        self.chirp_var.set("0")
        self.chirp_var.pack(side="left", padx=4)

        self.show_chirp = BooleanVar(value=True)
        self.show_avg = BooleanVar(value=True)
        self.show_chirp_max = BooleanVar(value=True)
        self.show_frame_hold = BooleanVar(value=False)
        ttk.Checkbutton(opts, text="Chirp", variable=self.show_chirp).pack(side="left", padx=6)
        ttk.Checkbutton(opts, text="Avg", variable=self.show_avg).pack(side="left", padx=2)
        ttk.Checkbutton(opts, text="Chirp max", variable=self.show_chirp_max).pack(
            side="left", padx=2
        )
        ttk.Checkbutton(
            opts,
            text="Frame max hold",
            variable=self.show_frame_hold,
            command=self._on_frame_hold_toggle,
        ).pack(side="left", padx=2)
        ttk.Button(opts, text="Reset hold", command=self.reset_frame_hold).pack(
            side="left", padx=4
        )

        ttk.Separator(opts, orient="vertical").pack(side="left", fill="y", padx=8)
        ttk.Label(opts, text="X-axis:").pack(side="left")
        self.axis_mode = StringVar(value="meters")
        ttk.Radiobutton(
            opts, text="Range (m)", variable=self.axis_mode, value="meters"
        ).pack(side="left", padx=2)
        ttk.Radiobutton(
            opts, text="Freq (MHz)", variable=self.axis_mode, value="mhz"
        ).pack(side="left", padx=2)

        self.rx_sel = RxChannelSelector(opts)

        # Plot row: [TX power] [canvas] [RX gain]
        plot_host = tk.Frame(parent, bg=COLORS["bg"], highlightthickness=0, bd=0)
        plot_host.grid(row=1, column=0, sticky="nsew")
        plot_host.columnconfigure(1, weight=1)
        plot_host.rowconfigure(0, weight=1)

        tx_col = ttk.Frame(plot_host)
        tx_col.grid(row=0, column=0, sticky="ns", padx=(0, 4), pady=4)
        self._tx_panel = bordered_panel(tx_col, padx=4, pady=4)
        self._tx_panel.pack(side="top", fill="y", expand=True)
        tx_inner = panel_content(self._tx_panel)
        ttk.Label(tx_inner, text="TX", font=("", 8)).pack(side="top")
        self.tx_pwr_value_lbl = ttk.Label(tx_inner, text=f"{_TX_PWR_DB_MIN} dB", font=("", 8))
        self.tx_pwr_value_lbl.pack(side="top")
        # Vertical Scale: from_ is top → put max at top.
        self.tx_pwr_var = tk.DoubleVar(value=float(_TX_PWR_DB_MIN))
        self.tx_pwr_scale = tk.Scale(
            tx_inner,
            from_=_TX_PWR_DB_MAX,
            to=_TX_PWR_DB_MIN,
            resolution=_TX_PWR_DB_STEP,
            orient=tk.VERTICAL,
            variable=self.tx_pwr_var,
            length=280,
            showvalue=0,
            command=self._on_tx_pwr_changed,
        )
        self.tx_pwr_scale.pack(side="top", fill="y", expand=True)
        style_tk_scale(self.tx_pwr_scale)
        ttk.Label(tx_inner, text="Pwr", font=("", 8)).pack(side="bottom")

        canvas_host = tk.Frame(plot_host, bg=COLORS["plot_bg"], highlightthickness=0, bd=0)
        canvas_host.grid(row=0, column=1, sticky="nsew")
        self.plot = EmbeddedFigure(canvas_host, figsize=(9.0, 4.2), toolbar=False)

        rx_col = ttk.Frame(plot_host)
        rx_col.grid(row=0, column=2, sticky="ns", padx=(4, 0), pady=4)
        self._rx_panel = bordered_panel(rx_col, padx=4, pady=4)
        self._rx_panel.pack(side="top", fill="y", expand=True)
        rx_inner = panel_content(self._rx_panel)
        ttk.Label(rx_inner, text="RX", font=("", 8)).pack(side="top")
        self.rx_gain_value_lbl = ttk.Label(
            rx_inner, text=f"{_RX_GAIN_DB_MIN} dB", font=("", 8)
        )
        self.rx_gain_value_lbl.pack(side="top")
        self.rx_gain_var = tk.DoubleVar(value=float(_RX_GAIN_DB_MIN))
        self.rx_gain_scale = tk.Scale(
            rx_inner,
            from_=_RX_GAIN_DB_MAX,
            to=_RX_GAIN_DB_MIN,
            resolution=_RX_GAIN_DB_STEP,
            orient=tk.VERTICAL,
            variable=self.rx_gain_var,
            length=280,
            showvalue=0,
            command=self._on_rx_gain_changed,
        )
        self.rx_gain_scale.pack(side="top", fill="y", expand=True)
        style_tk_scale(self.rx_gain_scale)
        ttk.Label(rx_inner, text="Gain", font=("", 8)).pack(side="bottom")

        footer = ttk.Frame(parent)
        footer.grid(row=2, column=0, sticky="ew")

        # Filter row: HPF (left) | Range res (center) | LPF (right)
        filters = ttk.Frame(footer)
        filters.pack(side="top", fill="x", pady=(4, 2))
        filters.columnconfigure(0, weight=0)
        filters.columnconfigure(1, weight=1)
        filters.columnconfigure(2, weight=0)

        # --- HPF ---
        self._hpf_panel = bordered_panel(filters)
        self._hpf_panel.grid(row=0, column=0, sticky="w")
        self._hpf_track = panel_content(self._hpf_panel)

        ends = ttk.Frame(self._hpf_track)
        ends.pack(side="top", fill="x")
        ttk.Label(ends, text=f"{_HPF_KHZ_MIN}", font=("", 8)).pack(side="left")
        self._hpf_value_lbl = ttk.Label(ends, text="HPF 0 kHz", font=("", 8))
        self._hpf_value_lbl.pack(side="left", expand=True, padx=4)
        ttk.Label(ends, text=f"{_HPF_KHZ_MAX} kHz", font=("", 8)).pack(side="right")

        self.hpf_var = tk.DoubleVar(value=float(_HPF_KHZ_MIN))
        self.hpf_scale = tk.Scale(
            self._hpf_track,
            from_=_HPF_KHZ_MIN,
            to=_HPF_KHZ_MAX,
            resolution=_HPF_KHZ_STEP,
            orient=tk.HORIZONTAL,
            variable=self.hpf_var,
            showvalue=0,
            length=200,
            command=self._on_hpf_changed,
        )
        self.hpf_scale.pack(side="top", fill="x")
        style_tk_scale(self.hpf_scale)
        self._hpf_status = ttk.Label(self._hpf_track, text="", font=("", 8))
        self._hpf_status.pack(side="top")

        # --- Range resolution (center) ---
        mid = ttk.Frame(filters)
        mid.grid(row=0, column=1, sticky="ew")
        mid.columnconfigure(0, weight=1)
        mid.columnconfigure(1, weight=0)
        mid.columnconfigure(2, weight=1)

        self._dr_panel = bordered_panel(mid)
        self._dr_panel.grid(row=0, column=1)
        self._dr_track = panel_content(self._dr_panel)

        dr_ends = ttk.Frame(self._dr_track)
        dr_ends.pack(side="top", fill="x")
        ttk.Label(dr_ends, text=f"{_DR_M_FINE:.2f}", font=("", 8)).pack(side="left")
        self._dr_value_lbl = ttk.Label(dr_ends, text="Range res —", font=("", 8))
        self._dr_value_lbl.pack(side="left", expand=True, padx=4)
        ttk.Label(dr_ends, text=f"{_DR_M_COARSE:.2f} m", font=("", 8)).pack(side="right")

        self.dr_var = tk.DoubleVar(value=0.15)
        self.dr_scale = tk.Scale(
            self._dr_track,
            from_=_DR_M_FINE,
            to=_DR_M_COARSE,
            resolution=_DR_M_STEP,
            orient=tk.HORIZONTAL,
            variable=self.dr_var,
            showvalue=0,
            length=200,
            command=self._on_chirp_bw_changed,
        )
        self.dr_scale.pack(side="top", fill="x")
        style_tk_scale(self.dr_scale)
        self._dr_status = ttk.Label(self._dr_track, text="", font=("", 8))
        self._dr_status.pack(side="top")

        # --- LPF ---
        self._lpf_panel = bordered_panel(filters)
        self._lpf_panel.grid(row=0, column=2, sticky="e")
        self._lpf_track = panel_content(self._lpf_panel)

        lpf_ends = ttk.Frame(self._lpf_track)
        lpf_ends.pack(side="top", fill="x")
        ttk.Label(lpf_ends, text=f"{_LPF_MHZ_MIN}", font=("", 8)).pack(side="left")
        self._lpf_value_lbl = ttk.Label(
            lpf_ends, text=f"LPF {_LPF_MHZ_MIN} MHz", font=("", 8)
        )
        self._lpf_value_lbl.pack(side="left", expand=True, padx=4)
        ttk.Label(lpf_ends, text=f"{_LPF_MHZ_MAX} MHz", font=("", 8)).pack(side="right")

        self.lpf_var = tk.DoubleVar(value=float(_LPF_MHZ_MIN))
        self.lpf_scale = tk.Scale(
            self._lpf_track,
            from_=_LPF_MHZ_MIN,
            to=_LPF_MHZ_MAX,
            resolution=_LPF_MHZ_STEP,
            orient=tk.HORIZONTAL,
            variable=self.lpf_var,
            showvalue=0,
            length=200,
            command=self._on_lpf_changed,
        )
        self.lpf_scale.pack(side="top", fill="x")
        style_tk_scale(self.lpf_scale)
        self._lpf_status = ttk.Label(self._lpf_track, text="", font=("", 8))
        self._lpf_status.pack(side="top")

        self.toolbar = NavigationToolbar2Tk(
            self.plot.canvas, footer, pack_toolbar=False
        )
        self.toolbar.pack(side="bottom", fill="x")
        self.toolbar.update()
        style_mpl_toolbar(self.toolbar)

        self.plot.canvas.mpl_connect("draw_event", self._on_plot_draw)
        self.plot.canvas.get_tk_widget().bind(
            "<Configure>", self._on_plot_configure, add="+"
        )

        self._frame_hold: dict[int, np.ndarray] = {}
        self._last_rx: list[int] = []
        self._hold_n_bins: int | None = None
        self._last_hpf_khz: int | None = None
        self._last_lpf_mhz: int | None = None
        self._last_tx_pwr_db: int | None = None
        self._last_rx_gain_db: int | None = None
        self._last_chirp_bw_hz: float | None = None
        parent.after_idle(self._push_rf_controls)

    def _hpf_khz(self) -> int:
        """Current HPF cutoff snapped to the slider step (kHz)."""
        try:
            raw = float(self.hpf_var.get())
        except (TypeError, ValueError, tk.TclError):
            return _HPF_KHZ_MIN
        stepped = int(round(raw / _HPF_KHZ_STEP) * _HPF_KHZ_STEP)
        return int(np.clip(stepped, _HPF_KHZ_MIN, _HPF_KHZ_MAX))

    def _on_hpf_changed(self, _value=None) -> None:
        """Notify the live device when the HPF cutoff changes."""
        khz = self._hpf_khz()
        self._hpf_value_lbl.configure(text=f"HPF {khz} kHz")
        if self._last_hpf_khz == khz:
            return
        self._last_hpf_khz = khz
        cutoff_hz = float(khz) * 1_000.0
        device = self.device
        if device is None:
            self._hpf_status.configure(text="(no device)")
            return
        try:
            device.updateHpf(cutoff_hz)
        except Exception as exc:  # noqa: BLE001
            self._hpf_status.configure(text=f"HPF error: {exc}")
            return
        self._hpf_status.configure(text="applied → device")

    def _lpf_mhz(self) -> int:
        """Current LPF cutoff snapped to the slider step (MHz)."""
        try:
            raw = float(self.lpf_var.get())
        except (TypeError, ValueError, tk.TclError):
            return _LPF_MHZ_MIN
        stepped = int(round(raw / _LPF_MHZ_STEP) * _LPF_MHZ_STEP)
        return int(np.clip(stepped, _LPF_MHZ_MIN, _LPF_MHZ_MAX))

    def _on_lpf_changed(self, _value=None) -> None:
        """Notify the live device when the LPF cutoff changes."""
        mhz = self._lpf_mhz()
        self._lpf_value_lbl.configure(text=f"LPF {mhz} MHz")
        if self._last_lpf_mhz == mhz:
            return
        self._last_lpf_mhz = mhz
        cutoff_hz = float(mhz) * 1_000_000.0
        device = self.device
        if device is None:
            self._lpf_status.configure(text="(no device)")
            return
        try:
            device.updateLpf(cutoff_hz)
        except Exception as exc:  # noqa: BLE001
            self._lpf_status.configure(text=f"LPF error: {exc}")
            return
        self._lpf_status.configure(text="applied → device")

    def _tx_pwr_db(self) -> int:
        try:
            raw = float(self.tx_pwr_var.get())
        except (TypeError, ValueError, tk.TclError):
            return _TX_PWR_DB_MIN
        stepped = int(round(raw / _TX_PWR_DB_STEP) * _TX_PWR_DB_STEP)
        return int(np.clip(stepped, _TX_PWR_DB_MIN, _TX_PWR_DB_MAX))

    def _rx_gain_db(self) -> int:
        try:
            raw = float(self.rx_gain_var.get())
        except (TypeError, ValueError, tk.TclError):
            return _RX_GAIN_DB_MIN
        return min(_RX_GAIN_DB_VALUES, key=lambda v: abs(v - raw))

    def _on_tx_pwr_changed(self, _value=None) -> None:
        """Notify the live device when TX power changes."""
        db = self._tx_pwr_db()
        self.tx_pwr_value_lbl.configure(text=f"{db} dB")
        if self._last_tx_pwr_db == db:
            return
        self._last_tx_pwr_db = db
        device = self.device
        if device is None:
            return
        try:
            device.updatePwr(float(db))
        except Exception:  # noqa: BLE001
            return

    def _on_rx_gain_changed(self, _value=None) -> None:
        """Notify the live device when RX gain changes."""
        db = self._rx_gain_db()
        self.rx_gain_value_lbl.configure(text=f"{db} dB")
        if self._last_rx_gain_db == db:
            return
        self._last_rx_gain_db = db
        device = self.device
        if device is None:
            return
        try:
            device.updateGain(float(db))
        except Exception:  # noqa: BLE001
            return

    def _push_rf_controls(self) -> None:
        """Push HPF / LPF / TX / RX / chirp BW to the current device."""
        self._sync_chirp_bw_slider_from_device()
        self._on_chirp_bw_changed()
        self._on_hpf_changed()
        self._on_lpf_changed()
        self._on_tx_pwr_changed()
        self._on_rx_gain_changed()

    def _dr_m_from_bw_hz(self, bandwidth_hz: float) -> float:
        """Convert chirp bandwidth (Hz) to snapped range resolution (m)."""
        bw = max(float(bandwidth_hz), 1.0)
        dr = SPEED_OF_LIGHT / (2.0 * bw)
        stepped = round(dr / _DR_M_STEP) * _DR_M_STEP
        return float(np.clip(stepped, _DR_M_FINE, _DR_M_COARSE))

    def _bw_hz_from_dr_m(self, range_res_m: float) -> float:
        """Convert range resolution (m) to chirp bandwidth (Hz)."""
        dr = max(float(range_res_m), _DR_M_FINE)
        bw = SPEED_OF_LIGHT / (2.0 * dr)
        return float(np.clip(bw, _CHIRP_BW_HZ_MIN, _CHIRP_BW_HZ_MAX))

    def _sync_chirp_bw_slider_from_device(self) -> None:
        """Initialize the range-resolution slider from the device chirp bandwidth."""
        device = self.device
        if device is None:
            return
        bw = float(getattr(device, "chirp_bandwidth_hz", device.config.bandwidth_hz))
        dr = self._dr_m_from_bw_hz(bw)
        self.dr_var.set(dr)
        self._dr_value_lbl.configure(text=f"{dr:.2f} m | {bw / 1e9:.2f} GHz")

    def _on_chirp_bw_changed(self, _value=None) -> None:
        """Map range-resolution slider → ``device.updateChirpBw``."""
        try:
            dr = float(self.dr_var.get())
        except (TypeError, ValueError, tk.TclError):
            return
        dr = float(np.clip(round(dr / _DR_M_STEP) * _DR_M_STEP, _DR_M_FINE, _DR_M_COARSE))
        bw = self._bw_hz_from_dr_m(dr)
        self._dr_value_lbl.configure(text=f"{dr:.2f} m | {bw / 1e9:.2f} GHz")
        if self._last_chirp_bw_hz is not None and abs(self._last_chirp_bw_hz - bw) < 1.0:
            return
        self._last_chirp_bw_hz = bw
        device = self.device
        if device is None:
            return
        try:
            device.updateChirpBw(bw)
        except Exception:  # noqa: BLE001
            return
        self.reset_frame_hold()

    def _on_plot_draw(self, _event=None) -> None:
        self._align_filter_sliders()

    def _on_plot_configure(self, _event=None) -> None:
        self._align_filter_sliders()

    def _align_filter_sliders(self) -> None:
        """Align HPF (left) and LPF (right) under the plot x-axis ends."""
        canvas_widget = self.plot.canvas.get_tk_widget()
        canvas_w = canvas_widget.winfo_width()
        if canvas_w <= 1 or self.frame is None:
            return
        pos = self.plot.axes.get_position()
        try:
            page_left = self.frame.winfo_rootx()
            page_right = page_left + self.frame.winfo_width()
            axes_left = (
                canvas_widget.winfo_rootx()
                - page_left
                + int(round(pos.x0 * canvas_w))
            )
            axes_right = (
                canvas_widget.winfo_rootx()
                - page_left
                + int(round(pos.x1 * canvas_w))
            )
            right_pad = max(0, page_right - page_left - axes_right)
        except tk.TclError:
            axes_left = int(round(pos.x0 * canvas_w))
            right_pad = int(round((1.0 - pos.x1) * canvas_w))
        self._hpf_panel.grid_configure(padx=(max(0, axes_left), 0))
        self._lpf_panel.grid_configure(padx=(0, max(0, right_pad)))

    def _range_at_freq_hz(self, frame: RadarFrame, freq_hz: float) -> float:
        """Convert beat frequency (Hz) to range (m) for the current chirp slope."""
        return float(freq_hz * SPEED_OF_LIGHT / (2.0 * frame.config.chirp_slope))

    def on_device_changed(self) -> None:
        """Re-apply RF controls to the newly selected device."""
        self._last_hpf_khz = None
        self._last_lpf_mhz = None
        self._last_tx_pwr_db = None
        self._last_rx_gain_db = None
        self._last_chirp_bw_hz = None
        self._push_rf_controls()

    def reset_frame_hold(self) -> None:
        """Clear frame-to-frame max-hold profiles."""
        self._frame_hold.clear()
        self._hold_n_bins = None

    def _on_frame_hold_toggle(self) -> None:
        if self.show_frame_hold.get():
            self.reset_frame_hold()

    def _update_frame_hold(self, mag: np.ndarray, rx_list: list[int]) -> None:
        """Accumulate max |Range FFT| per range bin across frames (chirp-max seed)."""
        n_bins = mag.shape[0]
        if self._hold_n_bins is not None and self._hold_n_bins != n_bins:
            self.reset_frame_hold()
        self._hold_n_bins = n_bins

        for rx in rx_list:
            frame_profile = mag[:, :, rx].max(axis=1)
            held = self._frame_hold.get(rx)
            if held is None or held.shape != frame_profile.shape:
                self._frame_hold[rx] = frame_profile.astype(np.float64, copy=True)
            else:
                np.maximum(held, frame_profile, out=held)

    def update(self, frame: RadarFrame) -> None:
        if not self.is_visible:
            return
        self.rx_sel.sync(frame.cube.shape[2])
        rx_list = self.rx_sel.selected()
        if rx_list != self._last_rx:
            self.reset_frame_hold()
            self._last_rx = list(rx_list)

        try:
            chirp_idx = int(self.chirp_var.get())
        except ValueError:
            chirp_idx = 0
        chirp_idx = int(np.clip(chirp_idx, 0, frame.cube.shape[1] - 1))

        mag = np.abs(frame.range_cube)
        if self.show_frame_hold.get():
            self._update_frame_hold(mag, rx_list)

        hpf_khz = self._hpf_khz()
        hpf_hz = hpf_khz * 1_000.0
        lpf_mhz = self._lpf_mhz()
        lpf_hz = lpf_mhz * 1_000_000.0
        use_mhz = self.axis_mode.get() == "mhz"

        if use_mhz:
            x_axis = frame.beat_freq_axis_mhz()
            x_label = "Beat frequency (MHz)"
            hpf_x = hpf_khz / 1000.0
            lpf_x = float(lpf_mhz)
        else:
            x_axis = frame.range_axis()
            x_label = "Range (m)"
            hpf_x = self._range_at_freq_hz(frame, hpf_hz)
            lpf_x = self._range_at_freq_hz(frame, lpf_hz)

        ax = self.plot.axes
        ax.clear()

        styles = {
            "chirp": "-",
            "avg": "--",
            "chirp_max": ":",
            "frame_hold": "-.",
        }
        for rx in rx_list:
            if self.show_chirp.get():
                profile = mag[:, chirp_idx, rx]
                ax.plot(
                    x_axis,
                    20 * np.log10(profile + 1e-12),
                    styles["chirp"],
                    lw=0.9,
                    label=f"RX {rx} (chirp {chirp_idx})",
                )
            if self.show_avg.get():
                profile = mag[:, :, rx].mean(axis=1)
                ax.plot(
                    x_axis,
                    20 * np.log10(profile + 1e-12),
                    styles["avg"],
                    lw=1.1,
                    label=f"RX {rx} (avg)",
                )
            if self.show_chirp_max.get():
                profile = mag[:, :, rx].max(axis=1)
                ax.plot(
                    x_axis,
                    20 * np.log10(profile + 1e-12),
                    styles["chirp_max"],
                    lw=1.1,
                    label=f"RX {rx} (chirp max)",
                )
            if self.show_frame_hold.get() and rx in self._frame_hold:
                profile = self._frame_hold[rx]
                ax.plot(
                    x_axis,
                    20 * np.log10(profile + 1e-12),
                    styles["frame_hold"],
                    lw=1.4,
                    label=f"RX {rx} (frame hold)",
                )

        ax.axvline(
            hpf_x,
            color="#e07a7a",
            ls="--",
            lw=1.2,
            alpha=0.9,
            label=f"HPF {hpf_khz} kHz",
        )
        ax.axvline(
            lpf_x,
            color="#7dcea0",
            ls="--",
            lw=1.2,
            alpha=0.9,
            label=f"LPF {lpf_mhz} MHz",
        )
        ax.set_xlabel(x_label)
        ax.set_ylabel("Magnitude (dB)")
        hold_tag = " + frame hold" if self.show_frame_hold.get() else ""
        ax.set_title(
            f"Range FFT ({self.rx_sel.label()}{hold_tag})  |  frame {frame.frame_id}"
        )
        ax.grid(True, alpha=0.3)
        if rx_list:
            ax.legend(loc="upper right", fontsize=7, ncol=2)
        self.plot.draw_idle()
