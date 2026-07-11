"""Matplotlib and Plotly visualization for FMCW radar data."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

try:
    import plotly.graph_objects as go
except ImportError:  # pragma: no cover
    go = None

from .config import RadarConfig
from .process import RadarProcessor
from .testing import QAReport

_LAYOUT = dict(
    template="plotly_white",
    hovermode="x unified",
    legend=dict(title="Click to show/hide", itemclick="toggle", itemdoubleclick="toggleothers"),
    margin=dict(l=60, r=30, t=50, b=50),
)


def _require_plotly():
    if go is None:
        raise ImportError("Plotly is required. Install with: pip install plotly")


class MatplotlibRadarPlotter:
    """Static matplotlib plots for FMCW radar data."""

    def __init__(self, config: RadarConfig) -> None:
        self.config = config
        self.processor = RadarProcessor(config)

    def plot_raw_adc_chirp(
        self,
        cube: np.ndarray,
        *,
        rx_idx: int | None = None,
        chirp_idx: int = 0,
        ax: plt.Axes | None = None,
    ) -> plt.Axes:
        config = self.config
        t_us = np.arange(cube.shape[0]) / config.adc_sample_rate_hz * 1e6
        rx_indices = range(cube.shape[2]) if rx_idx is None else [rx_idx]
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 4))
        for idx in rx_indices:
            samples = np.asarray(cube[:, chirp_idx, idx], dtype=np.float32)
            ax.plot(t_us, samples, linewidth=0.9, label=f"RX {idx}")
        ax.set_xlabel("Time (µs)")
        ax.set_ylabel("ADC counts")
        if rx_idx is None:
            ax.set_title(f"Raw ADC — All RX, Chirp {chirp_idx}")
            ax.legend(loc="upper right", ncol=min(cube.shape[2], 4))
        else:
            ax.set_title(f"Raw ADC — RX {rx_idx}, Chirp {chirp_idx}")
        ax.grid(True, alpha=0.3)
        return ax

    def plot_range_profile(
        self,
        cube: np.ndarray,
        *,
        rx_idx: int = 0,
        chirp_idx: int = 0,
        ax: plt.Axes | None = None,
    ) -> plt.Axes:
        r_cube = self.processor.range_fft(cube)
        profile = np.abs(r_cube[:, chirp_idx, rx_idx])
        range_axis = self.processor.compute_range_axis(r_cube.shape[0])
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 4))
        ax.plot(range_axis, 20 * np.log10(profile + 1e-12))
        ax.set_xlabel("Range (m)")
        ax.set_ylabel("Magnitude (dB)")
        ax.set_title(f"Range Profile — RX {rx_idx}, Chirp {chirp_idx}")
        ax.grid(True, alpha=0.3)
        return ax

    @staticmethod
    def plot_range_doppler(
        rd_map: np.ndarray,
        range_axis: np.ndarray,
        doppler_axis: np.ndarray,
        *,
        title: str = "Range-Doppler Map",
        ax: plt.Axes | None = None,
    ) -> plt.Axes:
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 6))
        rd_db = 20 * np.log10(rd_map + 1e-12)
        extent = [doppler_axis[0], doppler_axis[-1], range_axis[-1], range_axis[0]]
        im = ax.imshow(rd_db, aspect="auto", extent=extent, cmap="viridis")
        ax.set_xlabel("Velocity (m/s)")
        ax.set_ylabel("Range (m)")
        ax.set_title(title)
        plt.colorbar(im, ax=ax, label="dB")
        return ax

    def plot_detections(
        self,
        rd_map: np.ndarray,
        mask: np.ndarray,
        range_axis: np.ndarray,
        doppler_axis: np.ndarray,
        *,
        ax: plt.Axes | None = None,
    ) -> plt.Axes:
        ax = self.plot_range_doppler(
            rd_map, range_axis, doppler_axis, title="CFAR Detections", ax=ax
        )
        ri, di = np.where(mask)
        ax.scatter(doppler_axis[di], range_axis[ri], c="red", s=20, marker="x", label="Detections")
        ax.legend()
        return ax

    def plot_angle_spectrum(
        self,
        rd_cube: np.ndarray,
        range_idx: int,
        doppler_idx: int,
        *,
        ax: plt.Axes | None = None,
    ) -> plt.Axes:
        angle_cube = self.processor.angle_fft(rd_cube)
        spectrum = np.abs(angle_cube[range_idx, doppler_idx, :])
        angle_axis = self.processor.compute_angle_axis(angle_cube.shape[2])
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 4))
        ax.plot(angle_axis, 20 * np.log10(spectrum + 1e-12))
        ax.set_xlabel("Azimuth (deg)")
        ax.set_ylabel("Magnitude (dB)")
        ax.set_title(f"Angle Spectrum — Range bin {range_idx}, Doppler bin {doppler_idx}")
        ax.grid(True, alpha=0.3)
        return ax


class PlotlyRadarPlotter:
    """Interactive Plotly charts for FMCW radar data."""

    def __init__(self, config: RadarConfig) -> None:
        self.config = config
        self.processor = RadarProcessor(config)

    def raw_adc_chirp(
        self,
        cube: np.ndarray,
        *,
        chirp_idx: int = 0,
        rx_idx: int | None = None,
    ) -> "go.Figure":
        _require_plotly()
        t_us = np.arange(cube.shape[0]) / self.config.adc_sample_rate_hz * 1e6
        rx_indices = range(cube.shape[2]) if rx_idx is None else [rx_idx]
        fig = go.Figure()
        for idx in rx_indices:
            fig.add_trace(
                go.Scatter(
                    x=t_us,
                    y=cube[:, chirp_idx, idx],
                    mode="lines",
                    name=f"RX {idx}",
                    line=dict(width=1.2),
                )
            )
        title = (
            f"Raw ADC — All RX, Chirp {chirp_idx}"
            if rx_idx is None
            else f"Raw ADC — RX {rx_idx}, Chirp {chirp_idx}"
        )
        fig.update_layout(title=title, xaxis_title="Time (µs)", yaxis_title="ADC counts", **_LAYOUT)
        return fig

    def range_profile(
        self,
        cube: np.ndarray,
        *,
        chirp_idx: int = 0,
        all_rx: bool = True,
    ) -> "go.Figure":
        _require_plotly()
        r_cube = self.processor.range_fft(cube)
        range_axis = self.processor.compute_range_axis(r_cube.shape[0])
        fig = go.Figure()
        rx_indices = range(cube.shape[2]) if all_rx else [0]
        for idx in rx_indices:
            profile_db = 20 * np.log10(np.abs(r_cube[:, chirp_idx, idx]) + 1e-12)
            fig.add_trace(
                go.Scatter(
                    x=range_axis,
                    y=profile_db,
                    mode="lines",
                    name=f"RX {idx}",
                    line=dict(width=1.2),
                )
            )
        fig.update_layout(
            title=f"Range Profile — Chirp {chirp_idx}",
            xaxis_title="Range (m)",
            yaxis_title="Magnitude (dB)",
            **_LAYOUT,
        )
        return fig

    @staticmethod
    def range_doppler(
        rd_map: np.ndarray,
        range_axis: np.ndarray,
        doppler_axis: np.ndarray,
        *,
        title: str = "Range-Doppler Map",
    ) -> "go.Figure":
        _require_plotly()
        rd_db = 20 * np.log10(rd_map + 1e-12)
        fig = go.Figure(
            data=go.Heatmap(
                x=doppler_axis,
                y=range_axis,
                z=rd_db,
                colorscale="Viridis",
                colorbar=dict(title="dB"),
            )
        )
        fig.update_layout(
            title=title,
            xaxis_title="Velocity (m/s)",
            yaxis_title="Range (m)",
            template="plotly_white",
            margin=dict(l=60, r=30, t=50, b=50),
        )
        fig.update_yaxes(autorange="reversed")
        return fig

    @classmethod
    def detections(
        cls,
        rd_map: np.ndarray,
        mask: np.ndarray,
        range_axis: np.ndarray,
        doppler_axis: np.ndarray,
    ) -> "go.Figure":
        _require_plotly()
        fig = cls.range_doppler(rd_map, range_axis, doppler_axis, title="CFAR Detections")
        ri, di = np.where(mask)
        fig.add_trace(
            go.Scatter(
                x=doppler_axis[di],
                y=range_axis[ri],
                mode="markers",
                name="Detections",
                marker=dict(color="red", size=7, symbol="x"),
            )
        )
        fig.update_layout(**_LAYOUT)
        return fig

    def angle_spectrum(
        self,
        rd_cube: np.ndarray,
        range_idx: int,
        doppler_idx: int,
    ) -> "go.Figure":
        _require_plotly()
        angle_cube = self.processor.angle_fft(rd_cube)
        spectrum_db = 20 * np.log10(np.abs(angle_cube[range_idx, doppler_idx, :]) + 1e-12)
        angle_axis = self.processor.compute_angle_axis(angle_cube.shape[2])
        fig = go.Figure(
            data=go.Scatter(
                x=angle_axis,
                y=spectrum_db,
                mode="lines",
                name="Angle spectrum",
                line=dict(width=1.5),
            )
        )
        fig.update_layout(
            title=f"Angle Spectrum — Range bin {range_idx}, Doppler bin {doppler_idx}",
            xaxis_title="Azimuth (deg)",
            yaxis_title="Magnitude (dB)",
            **_LAYOUT,
        )
        return fig

    @staticmethod
    def qa_rx_power(report: QAReport) -> "go.Figure":
        _require_plotly()
        labels = [f"RX {i}" for i in range(len(report.rx_power_db))]
        fig = go.Figure(data=go.Bar(x=labels, y=report.rx_power_db))
        fig.update_layout(
            title="RX power balance (0 dB = strongest)",
            xaxis_title="Receiver",
            yaxis_title="Relative power (dB)",
            template="plotly_white",
        )
        return fig

    @staticmethod
    def qa_chirp_rms(report: QAReport) -> "go.Figure":
        _require_plotly()
        rms = report.chirp_rms_per_rx
        if rms.size == 0:
            rms = report.chirp_rms[:, None]
        chirp_idx = np.arange(rms.shape[0])
        fig = go.Figure()
        for rx in range(rms.shape[1]):
            fig.add_trace(
                go.Scatter(
                    x=chirp_idx,
                    y=rms[:, rx],
                    mode="lines+markers",
                    name=f"RX {rx}",
                    line=dict(width=1.2),
                )
            )
        fig.update_layout(
            title="Chirp RMS stability (all RX)",
            xaxis_title="Chirp index",
            yaxis_title="RMS (ADC counts)",
            **_LAYOUT,
        )
        return fig

    def qa_adc_spectrum(
        self,
        cube: np.ndarray,
        *,
        chirp_idx: int = 0,
        rx_idx: int = 0,
    ) -> "go.Figure":
        _require_plotly()
        mini = cube[:, chirp_idx : chirp_idx + 1, rx_idx : rx_idx + 1]
        r_line = self.processor.range_fft(mini)
        spec_db = 20 * np.log10(np.abs(r_line[:, 0, 0]) + 1e-12)
        range_axis = self.processor.compute_range_axis(r_line.shape[0])
        fig = go.Figure(
            data=go.Scatter(x=range_axis, y=spec_db, mode="lines", name=f"RX {rx_idx}")
        )
        fig.update_layout(
            title=f"Beat spectrum (rfft) — RX {rx_idx}, Chirp {chirp_idx}",
            xaxis_title="Equivalent range (m)",
            yaxis_title="Magnitude (dB)",
            template="plotly_white",
        )
        return fig

    @staticmethod
    def qa_rx_correlation(
        report: QAReport,
        *,
        ref_rx: int = 0,
        threshold: float | None = 0.5,
    ) -> "go.Figure":
        _require_plotly()
        corr = report.rx_correlation
        labels = [f"RX {i}" for i in range(len(corr))]
        colors = ["#636EFA" if i == ref_rx else "#00CC96" for i in range(len(corr))]
        fig = go.Figure(
            data=go.Bar(
                x=labels,
                y=corr,
                marker_color=colors,
                text=[f"{v:.3f}" for v in corr],
                textposition="outside",
            )
        )
        if threshold is not None:
            fig.add_hline(
                y=threshold,
                line_dash="dash",
                line_color="gray",
                annotation_text=f"limit ({threshold:.2f})",
                annotation_position="bottom right",
            )
        fig.update_layout(
            title=f"RX correlation vs RX {ref_rx} (peak normalized xcorr)",
            xaxis_title="Receiver",
            yaxis_title="Peak |xcorr| vs reference",
            yaxis=dict(range=[0, 1.05]),
            template="plotly_white",
        )
        return fig

    @staticmethod
    def qa_rx_correlation_matrix(report: QAReport) -> "go.Figure":
        _require_plotly()
        corr = report.rx_correlation_matrix
        labels = [f"RX {i}" for i in range(corr.shape[0])]
        fig = go.Figure(
            data=go.Heatmap(
                x=labels,
                y=labels,
                z=corr,
                zmin=0.0,
                zmax=1.0,
                colorscale="Viridis",
                colorbar=dict(title="Peak |xcorr|"),
                text=np.round(corr, 3),
                texttemplate="%{text}",
                textfont=dict(size=12),
            )
        )
        fig.update_layout(
            title="RX correlation matrix (peak normalized xcorr)",
            xaxis_title="Receiver",
            yaxis_title="Receiver",
            template="plotly_white",
        )
        return fig

    @staticmethod
    def show(fig: "go.Figure") -> None:
        fig.show()


# --- Backward-compatible functions ---

def plot_raw_adc_chirp(cube: np.ndarray, config: RadarConfig, **kwargs) -> plt.Axes:
    return MatplotlibRadarPlotter(config).plot_raw_adc_chirp(cube, **kwargs)


def plot_range_profile(cube: np.ndarray, config: RadarConfig, **kwargs) -> plt.Axes:
    return MatplotlibRadarPlotter(config).plot_range_profile(cube, **kwargs)


def plot_range_doppler(rd_map, range_axis, doppler_axis, **kwargs) -> plt.Axes:
    return MatplotlibRadarPlotter.plot_range_doppler(rd_map, range_axis, doppler_axis, **kwargs)


def plot_detections(rd_map, mask, range_axis, doppler_axis, **kwargs) -> plt.Axes:
    ax = plot_range_doppler(rd_map, range_axis, doppler_axis, title="CFAR Detections", **kwargs)
    ri, di = np.where(mask)
    ax.scatter(doppler_axis[di], range_axis[ri], c="red", s=20, marker="x", label="Detections")
    ax.legend()
    return ax


def plot_angle_spectrum(rd_cube, config, range_idx, doppler_idx, **kwargs) -> plt.Axes:
    return MatplotlibRadarPlotter(config).plot_angle_spectrum(
        rd_cube, range_idx, doppler_idx, **kwargs
    )


def iplot_raw_adc_chirp(cube: np.ndarray, config: RadarConfig, **kwargs) -> "go.Figure":
    return PlotlyRadarPlotter(config).raw_adc_chirp(cube, **kwargs)


def iplot_range_profile(cube: np.ndarray, config: RadarConfig, **kwargs) -> "go.Figure":
    return PlotlyRadarPlotter(config).range_profile(cube, **kwargs)


def iplot_range_doppler(rd_map, range_axis, doppler_axis, **kwargs) -> "go.Figure":
    return PlotlyRadarPlotter.range_doppler(rd_map, range_axis, doppler_axis, **kwargs)


def iplot_detections(rd_map, mask, range_axis, doppler_axis) -> "go.Figure":
    return PlotlyRadarPlotter.detections(rd_map, mask, range_axis, doppler_axis)


def iplot_angle_spectrum(rd_cube, config, range_idx, doppler_idx) -> "go.Figure":
    return PlotlyRadarPlotter(config).angle_spectrum(rd_cube, range_idx, doppler_idx)


def iplot_qa_rx_power(report) -> "go.Figure":
    return PlotlyRadarPlotter.qa_rx_power(report)


def iplot_qa_chirp_rms(report) -> "go.Figure":
    return PlotlyRadarPlotter.qa_chirp_rms(report)


def iplot_qa_adc_spectrum(cube, config, **kwargs) -> "go.Figure":
    return PlotlyRadarPlotter(config).qa_adc_spectrum(cube, **kwargs)


def iplot_qa_rx_correlation(report, **kwargs) -> "go.Figure":
    return PlotlyRadarPlotter.qa_rx_correlation(report, **kwargs)


def iplot_qa_rx_correlation_matrix(report) -> "go.Figure":
    return PlotlyRadarPlotter.qa_rx_correlation_matrix(report)


show = PlotlyRadarPlotter.show
ishow = show
