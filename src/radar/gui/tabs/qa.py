"""Raw ADC QA measurement tab."""

from __future__ import annotations

from tkinter import ttk

import numpy as np

from ..frame import RadarFrame
from .base import MeasurementTab, register_tab
from .plotting import EmbeddedFigure, RxChannelSelector


@register_tab
class QaTab(MeasurementTab):
    """Live raw ADC QA summary filtered to selected RX channels."""

    title = "QA"
    order = 50

    def build(self, parent: ttk.Frame) -> None:
        opts = ttk.Frame(parent)
        opts.pack(side="top", fill="x", pady=(0, 4))
        self.rx_sel = RxChannelSelector(opts)

        top = ttk.Frame(parent)
        top.pack(side="top", fill="both", expand=False)

        self.summary = ttk.Label(
            top, text="Waiting for frames…", justify="left", font=("Consolas", 9)
        )
        self.summary.pack(side="left", fill="both", expand=True, padx=(0, 8))

        plots = ttk.Frame(parent)
        plots.pack(side="top", fill="both", expand=True)

        left = ttk.Frame(plots)
        left.pack(side="left", fill="both", expand=True)
        right = ttk.Frame(plots)
        right.pack(side="left", fill="both", expand=True)

        self.power_plot = EmbeddedFigure(left, figsize=(4.5, 3.2), toolbar=False)
        self.corr_plot = EmbeddedFigure(right, figsize=(4.5, 3.2), toolbar=False)

    def update(self, frame: RadarFrame) -> None:
        if not self.is_visible:
            return
        self.rx_sel.sync(frame.cube.shape[2])
        rx_list = self.rx_sel.selected()
        report = frame.qa_report

        # Filter summary lines that mention RX indices when possible
        self.summary.configure(
            text=report.summary() + f"\n\nPlotting: {self.rx_sel.label()}"
        )

        # RX power bars (selected only)
        axp = self.power_plot.axes
        axp.clear()
        if rx_list and report.rx_power_db.size:
            labels = [f"RX{i}" for i in rx_list]
            vals = report.rx_power_db[rx_list]
            axp.bar(labels, vals, color="#2a9d8f")
        axp.set_ylabel("Relative power (dB)")
        axp.set_title(f"RX power ({self.rx_sel.label()})")
        axp.grid(True, axis="y", alpha=0.3)
        self.power_plot.draw_idle()

        # Correlation vs RX0 + group delay (selected only)
        axc = self.corr_plot.axes
        axc.clear()
        if rx_list and report.rx_correlation.size:
            corr = report.rx_correlation[rx_list]
            delay = (
                report.rx_group_delay_samples[rx_list]
                if report.rx_group_delay_samples.size == report.rx_correlation.size
                else np.array([])
            )
            x = np.arange(len(rx_list))
            bars = axc.bar(x, corr, color="#457b9d")
            axc.set_xticks(x)
            axc.set_xticklabels([f"RX{i}" for i in rx_list])
            axc.set_ylim(0, 1.05)
            axc.axhline(
                frame.qa_thresholds.min_rx_correlation,
                color="gray",
                ls="--",
                lw=1,
            )
            if delay.size == corr.size:
                for bar, c, d in zip(bars, corr, delay):
                    axc.text(
                        bar.get_x() + bar.get_width() / 2,
                        min(float(c) + 0.03, 1.0),
                        f"{d:+.0f}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )
        axc.set_ylabel("Peak |xcorr| vs RX0")
        axc.set_title(f"RX correlation ({self.rx_sel.label()})")
        axc.grid(True, axis="y", alpha=0.3)
        self.corr_plot.draw_idle()
