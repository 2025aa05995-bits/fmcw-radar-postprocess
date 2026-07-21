"""Raw ADC time-domain measurement tab."""

from __future__ import annotations

import tkinter as tk
from tkinter import StringVar, ttk

import numpy as np

from ..frame import RadarFrame
from .base import MeasurementTab, register_tab
from .plotting import EmbeddedFigure, RxChannelSelector

_SAMPLE_OPTIONS = (128, 512, 1024, 2048, 4096)
# (label, Hz)
_RATE_OPTIONS: tuple[tuple[str, float], ...] = (
    ("20 MSPS", 20e6),
    ("40 MSPS", 40e6),
    ("80 MSPS", 80e6),
)
_RATE_BY_LABEL = {label: hz for label, hz in _RATE_OPTIONS}
_RATE_LABELS = [label for label, _ in _RATE_OPTIONS]


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

        ttk.Separator(opts, orient="vertical").pack(side="left", fill="y", padx=8)
        ttk.Label(opts, text="Samples:").pack(side="left")
        self.samples_var = StringVar(value=str(_SAMPLE_OPTIONS[2]))  # 1024 default
        self.samples_combo = ttk.Combobox(
            opts,
            textvariable=self.samples_var,
            values=[str(v) for v in _SAMPLE_OPTIONS],
            state="readonly",
            width=6,
        )
        self.samples_combo.pack(side="left", padx=4)
        self.samples_combo.bind("<<ComboboxSelected>>", self._on_samples_changed)

        ttk.Separator(opts, orient="vertical").pack(side="left", fill="y", padx=8)
        ttk.Label(opts, text="Sample rate:").pack(side="left")
        self.rate_var = StringVar(value="40 MSPS")
        self.rate_combo = ttk.Combobox(
            opts,
            textvariable=self.rate_var,
            values=_RATE_LABELS,
            state="readonly",
            width=8,
        )
        self.rate_combo.pack(side="left", padx=4)
        self.rate_combo.bind("<<ComboboxSelected>>", self._on_rate_changed)

        self.rx_sel = RxChannelSelector(opts)

        self.plot = EmbeddedFigure(parent, figsize=(9.0, 4.2))
        self._lines: list = []
        self._last_rx: list[int] = []
        self._last_n_samples: int | None = None
        self._last_fs_hz: float | None = None
        parent.after_idle(self._sync_controls_from_device)

    def _snap_samples(self, n: int) -> int:
        """Snap ``n`` to the nearest allowed dropdown value."""
        return min(_SAMPLE_OPTIONS, key=lambda v: abs(v - int(n)))

    def _snap_rate(self, fs_hz: float) -> tuple[str, float]:
        """Snap ``fs_hz`` to the nearest allowed rate option."""
        return min(_RATE_OPTIONS, key=lambda item: abs(item[1] - float(fs_hz)))

    def _set_samples_ui(self, n: int) -> None:
        self._suppress_setting_events = True
        try:
            self.samples_var.set(str(int(n)))
        finally:
            self._suppress_setting_events = False

    def _set_rate_ui(self, fs_hz: float) -> None:
        label, _ = self._snap_rate(fs_hz)
        self._suppress_setting_events = True
        try:
            self.rate_var.set(label)
        finally:
            self._suppress_setting_events = False

    def _sync_controls_from_device(self) -> None:
        """Initialize Samples / Sample-rate dropdowns from the live device config."""
        device = self.device
        if device is None:
            return

        n = self._snap_samples(
            int(getattr(device, "num_samples", device.config.num_samples))
        )
        self._set_samples_ui(n)

        fs = float(
            getattr(device, "adc_sample_rate_hz", device.config.adc_sample_rate_hz)
        )
        label, snapped_fs = self._snap_rate(fs)
        self._set_rate_ui(snapped_fs)

        ctrl = self.settings
        if ctrl is not None:
            ctrl.seed("num_samples", n, label="Samples", unit="")
            ctrl.seed("sample_rate_hz", snapped_fs, label="Sample rate", unit="Hz")

    def _on_samples_changed(self, _event=None) -> None:
        """Queue sample count to the device (non-blocking)."""
        if self._suppress_setting_events:
            return
        try:
            n = int(self.samples_var.get())
        except (TypeError, ValueError):
            return
        if n not in _SAMPLE_OPTIONS:
            n = self._snap_samples(n)
            self._set_samples_ui(n)

        ctrl = self.settings
        if ctrl is None:
            return

        def _revert(committed) -> None:
            self._set_samples_ui(int(committed))

        def _applied(_v) -> None:
            self._lines = []
            self._last_n_samples = None

        ctrl.request(
            "num_samples",
            n,
            label="Samples",
            apply=lambda d, v=n: d.updateSamples(v),
            equal=lambda a, b: int(a) == int(b),
            on_applied=_applied,
            on_discarded=_revert,
            on_error=lambda _e: _revert(ctrl.committed("num_samples", n)),
        )

    def _on_rate_changed(self, _event=None) -> None:
        """Queue sample rate to the device (non-blocking)."""
        if self._suppress_setting_events:
            return
        label = self.rate_var.get()
        fs = _RATE_BY_LABEL.get(label)
        if fs is None:
            label, fs = self._snap_rate(40e6)
            self._set_rate_ui(fs)

        ctrl = self.settings
        if ctrl is None:
            return

        def _revert(committed) -> None:
            self._set_rate_ui(float(committed))

        def _applied(_v) -> None:
            self._lines = []
            self._last_fs_hz = None

        ctrl.request(
            "sample_rate_hz",
            fs,
            label="Sample rate",
            unit="Hz",
            apply=lambda d, v=fs: d.updateSampleRate(v),
            on_applied=_applied,
            on_discarded=_revert,
            on_error=lambda _e: _revert(ctrl.committed("sample_rate_hz", fs)),
        )

    def on_device_changed(self) -> None:
        """Re-sync samples / rate from the newly selected device."""
        self._sync_controls_from_device()

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

        n_samples = int(frame.cube.shape[0])
        fs = float(frame.config.adc_sample_rate_hz)
        t_us = np.arange(n_samples) / fs * 1e6
        ax = self.plot.axes
        rebuild = (
            self._lines == []
            or self._last_rx != rx_list
            or self._last_n_samples != n_samples
            or self._last_fs_hz != fs
        )
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
            self._last_n_samples = n_samples
            self._last_fs_hz = fs
        else:
            for ln, rx in zip(self._lines, rx_list):
                ln.set_data(t_us, frame.cube[:, chirp_idx, rx])
            ax.relim()
            ax.autoscale_view()

        rate_label = self.rate_var.get() or f"{fs / 1e6:.0f} MSPS"
        ax.set_title(
            f"Raw ADC — chirp {chirp_idx}, N={n_samples}, {rate_label} "
            f"({self.rx_sel.label()})  |  frame {frame.frame_id}"
        )
        self.plot.draw_idle()
