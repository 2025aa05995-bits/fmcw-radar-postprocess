"""Continuous live FMCW radar GUI application."""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

from ..devices import RadarDeviceFactory
from ..testing import QAThresholds
from .live import LiveDataWorker
from .tabs import registered_tabs
from .theme import COLORS, apply_dark_theme


class RadarLiveApp:
    """
    Main window: continuous live fetch + notebook of measurement tabs.

    Tabs are discovered via ``@register_tab``. Live data comes from a
    ``RadarDevice`` created by ``RadarDeviceFactory`` (default: ``synthetic``).
    """

    def __init__(
        self,
        *,
        config_path: str | Path | None = None,
        device_type: str = "synthetic",
        interval_s: float = 0.15,
    ) -> None:
        self.root = tk.Tk()
        self.root.title("FMCW Radar — Live Measurements")
        self.root.minsize(960, 600)
        apply_dark_theme(self.root)
        # Start maximized (full desktop work area).
        try:
            self.root.state("zoomed")  # Windows
        except tk.TclError:
            self.root.attributes("-fullscreen", True)

        self.config_path = Path(config_path) if config_path else None
        self.device_type = device_type.strip().lower()

        device = RadarDeviceFactory.create(
            self.device_type,
            config_path=self.config_path,
        )
        self.worker = LiveDataWorker(
            device,
            interval_s=interval_s,
            qa_thresholds=QAThresholds(),
        )

        self._tabs: list = []
        self._active_tab = None
        self._running = False
        self._fps_count = 0
        self._fps_stamp = 0.0

        self._build_chrome()
        self._build_tabs()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_chrome(self) -> None:
        header = ttk.Frame(self.root, style="Header.TFrame", padding=(14, 10))
        header.pack(side="top", fill="x")

        brand = ttk.Frame(header, style="Header.TFrame")
        brand.pack(side="left")
        ttk.Label(brand, text="FMCW Radar", style="Brand.TLabel").pack(anchor="w")
        ttk.Label(
            brand,
            text="Live measurements",
            style="Subheader.TLabel",
        ).pack(anchor="w")

        toolbar = ttk.Frame(header, style="Header.TFrame")
        toolbar.pack(side="right")

        self.btn_start = ttk.Button(
            toolbar, text="Start", style="Accent.TButton", command=self.start
        )
        self.btn_start.pack(side="left", padx=(0, 4))
        self.btn_stop = ttk.Button(
            toolbar, text="Stop", style="Ghost.TButton", command=self.stop, state="disabled"
        )
        self.btn_stop.pack(side="left", padx=2)

        ttk.Separator(toolbar, orient="vertical").pack(side="left", fill="y", padx=10)

        ttk.Label(toolbar, text="Interval", style="HeaderMuted.TLabel").pack(side="left")
        ttk.Label(toolbar, text="(ms)", style="HeaderMuted.TLabel").pack(side="left", padx=(2, 4))
        self.interval_var = tk.StringVar(value="150")
        interval_entry = ttk.Entry(toolbar, textvariable=self.interval_var, width=6)
        interval_entry.pack(side="left", padx=(0, 4))
        ttk.Button(toolbar, text="Apply", command=self._apply_interval).pack(side="left", padx=2)

        ttk.Separator(toolbar, orient="vertical").pack(side="left", fill="y", padx=10)

        ttk.Label(toolbar, text="Device", style="HeaderMuted.TLabel").pack(side="left", padx=(0, 4))
        available = RadarDeviceFactory.available() or ["synthetic"]
        if self.device_type not in available:
            self.device_type = available[0]
        self.device_var = tk.StringVar(value=self.device_type)
        self.device_combo = ttk.Combobox(
            toolbar,
            textvariable=self.device_var,
            values=available,
            state="readonly",
            width=14,
        )
        self.device_combo.pack(side="left", padx=(0, 4))
        self.device_combo.bind("<<ComboboxSelected>>", self._on_device_selected)
        ttk.Button(toolbar, text="Apply device", command=self._apply_device).pack(
            side="left", padx=2
        )

        status_bar = ttk.Frame(self.root, style="Header.TFrame", padding=(14, 6))
        status_bar.pack(side="bottom", fill="x")
        self.status = ttk.Label(status_bar, text="Idle", style="Status.TLabel", anchor="w")
        self.status.pack(side="left", fill="x", expand=True)

        rule = tk.Frame(self.root, height=1, bg=COLORS["border"], bd=0, highlightthickness=0)
        rule.pack(side="top", fill="x")

        body = ttk.Frame(self.root, padding=(10, 8))
        body.pack(side="top", fill="both", expand=True)

        self.notebook = ttk.Notebook(body)
        self.notebook.pack(side="top", fill="both", expand=True)
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)

    def _build_tabs(self) -> None:
        for cls in registered_tabs():
            tab = cls()
            tab.bind_device(lambda: self.worker.device)
            tab.attach(self.notebook)
            self._tabs.append(tab)
        if self._tabs:
            self._tabs[0].on_show()
            self._active_tab = self._tabs[0]

    def _on_tab_changed(self, _event=None) -> None:
        try:
            idx = self.notebook.index(self.notebook.select())
        except tk.TclError:
            return
        if self._active_tab is not None:
            self._active_tab.on_hide()
        if 0 <= idx < len(self._tabs):
            self._active_tab = self._tabs[idx]
            self._active_tab.on_show()

    def _apply_interval(self) -> None:
        try:
            ms = float(self.interval_var.get())
        except ValueError:
            messagebox.showerror("Invalid interval", "Enter interval in milliseconds.")
            return
        self.worker.set_interval(ms / 1000.0)
        self.status.configure(text=f"Interval set to {ms:.0f} ms")

    def _on_device_selected(self, _event=None) -> None:
        self.status.configure(text=f"Device selected: {self.device_var.get()} (Apply to switch)")

    def _apply_device(self) -> None:
        name = self.device_var.get().strip().lower()
        try:
            device = RadarDeviceFactory.create(
                name,
                config_path=self.config_path,
            )
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Device error", str(exc))
            return

        was_running = self._running
        if was_running:
            self.stop()
        self.device_type = name
        self.worker.set_device(device)
        for tab in self._tabs:
            try:
                tab.on_device_changed()
            except Exception:  # noqa: BLE001
                pass
        cfg_note = (
            f"  |  config {device.config_path.name}"
            if device.config_path is not None
            else ""
        )
        self.status.configure(text=f"Device: {device.name}{cfg_note}")
        if was_running:
            self.start()

    def start(self) -> None:
        if self._running:
            return
        self._apply_interval()
        try:
            self.worker.start()
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Start failed", str(exc))
            return
        self._running = True
        self.btn_start.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self.status.configure(text=f"Running ({self.worker.device.name})…")
        self._poll()

    def stop(self) -> None:
        self._running = False
        self.worker.stop()
        self.btn_start.configure(state="normal")
        self.btn_stop.configure(state="disabled")
        self.status.configure(text="Stopped")

    def _poll(self) -> None:
        if not self._running:
            return
        frame = self.worker.get_latest()
        if frame is not None:
            for tab in self._tabs:
                try:
                    tab.ingest_frame(frame)
                except Exception:  # noqa: BLE001
                    pass

            if self._active_tab is not None:
                try:
                    self._active_tab.update(frame)
                except Exception as exc:  # noqa: BLE001
                    self.status.configure(text=f"Tab error: {exc}")
                else:
                    self._fps_count += 1
                    err = self.worker.last_error
                    if err:
                        self.status.configure(text=f"Device error: {err}")
                    else:
                        temp = (
                            f"  ·  T={frame.temperature_c:.1f}°C"
                            if frame.temperature_c is not None
                            else ""
                        )
                        self.status.configure(
                            text=(
                                f"{frame.source_name}  ·  frame {frame.frame_id}  ·  "
                                f"cube {frame.cube.shape}{temp}"
                            )
                        )
        elif self.worker.last_error:
            self.status.configure(text=f"Device error: {self.worker.last_error}")

        self.root.after(16, self._poll)

    def _on_close(self) -> None:
        self.stop()
        for tab in self._tabs:
            tab.teardown()
        self.root.destroy()

    def run(self) -> None:
        """Start live acquisition and enter the Tk main loop."""
        self.start()
        self.root.mainloop()


def launch(
    *,
    config_path: str | Path | None = None,
    device_type: str = "synthetic",
    interval_s: float = 0.15,
) -> None:
    """Convenience entry used by ``python -m radar.gui``."""
    app = RadarLiveApp(
        config_path=config_path,
        device_type=device_type,
        interval_s=interval_s,
    )
    app.run()
