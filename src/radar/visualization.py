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
    """Raise ImportError if Plotly is not installed."""
    if go is None:
        raise ImportError("Plotly is required. Install with: pip install plotly")


def _resolve_rx_indices(cube: np.ndarray, *, all_rx: bool, rx_idx: int | None) -> list[int]:
    """Return RX channel indices to plot."""
    if rx_idx is not None:
        if rx_idx < 0 or rx_idx >= cube.shape[2]:
            raise ValueError(f"rx_idx={rx_idx} out of range for {cube.shape[2]} RX channels")
        return [rx_idx]
    if all_rx:
        return list(range(cube.shape[2]))
    return [0]


def _iter_range_fft_profiles(
    r_cube: np.ndarray,
    *,
    rx_indices: list[int],
    chirp_idx: int,
    show_chirp: bool,
    show_avg: bool,
    show_max_hold: bool,
) -> list[tuple[str, np.ndarray, str]]:
    """
    Build linear-magnitude range profiles for plotting.

    Returns a list of ``(label, magnitude, linestyle)`` tuples.
    """
    if not (show_chirp or show_avg or show_max_hold):
        raise ValueError("Enable at least one of show_chirp, show_avg, or show_max_hold")

    mag = np.abs(r_cube)
    only_chirp = show_chirp and not show_avg and not show_max_hold
    traces: list[tuple[str, np.ndarray, str]] = []

    for rx in rx_indices:
        if show_chirp:
            label = f"RX {rx}" if only_chirp else f"RX {rx} (chirp {chirp_idx})"
            traces.append((label, mag[:, chirp_idx, rx], "solid"))
        if show_avg:
            traces.append((f"RX {rx} (avg)", mag[:, :, rx].mean(axis=1), "dash"))
        if show_max_hold:
            traces.append((f"RX {rx} (max hold)", mag[:, :, rx].max(axis=1), "dot"))
    return traces


def _range_profile_title(
    *,
    chirp_idx: int,
    show_chirp: bool,
    show_avg: bool,
    show_max_hold: bool,
) -> str:
    """Build a plot title from enabled range-profile modes."""
    parts: list[str] = []
    if show_chirp:
        parts.append(f"chirp {chirp_idx}")
    if show_avg:
        parts.append("avg")
    if show_max_hold:
        parts.append("max hold")
    return "Range FFT — " + ", ".join(parts)


_PLOTLY_DASH = {"solid": None, "dash": "dash", "dot": "dot"}
_MPL_LINESTYLE = {"solid": "-", "dash": "--", "dot": ":"}


class MatplotlibRadarPlotter:
    """Static matplotlib plots for FMCW radar data."""

    def __init__(self, config: RadarConfig) -> None:
        """Attach config and a ``RadarProcessor`` for FFT-based plots."""
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
        """
        Plot fast-time ADC samples for one or all RX channels (matplotlib).

        When ``rx_idx`` is None, overlays every RX with a legend.
        """
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
        rx_idx: int | None = None,
        chirp_idx: int = 0,
        all_rx: bool = True,
        show_chirp: bool = True,
        show_avg: bool = False,
        show_max_hold: bool = False,
        ax: plt.Axes | None = None,
    ) -> plt.Axes:
        """
        Plot range FFT magnitude in dB for each RX channel.

        Use ``show_chirp``, ``show_avg``, and ``show_max_hold`` to overlay a single
        chirp profile, chirp-averaged profile, and chirp max-hold profile.
        """
        r_cube = self.processor.range_fft(cube)
        range_axis = self.processor.compute_range_axis(r_cube.shape[0])
        rx_indices = _resolve_rx_indices(cube, all_rx=all_rx, rx_idx=rx_idx)
        traces = _iter_range_fft_profiles(
            r_cube,
            rx_indices=rx_indices,
            chirp_idx=chirp_idx,
            show_chirp=show_chirp,
            show_avg=show_avg,
            show_max_hold=show_max_hold,
        )
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 4))
        for label, profile, linestyle in traces:
            ax.plot(
                range_axis,
                20 * np.log10(profile + 1e-12),
                linewidth=0.9 if linestyle == "solid" else 1.1,
                linestyle=_MPL_LINESTYLE[linestyle],
                label=label,
            )
        ax.set_xlabel("Range (m)")
        ax.set_ylabel("Magnitude (dB)")
        ax.set_title(
            _range_profile_title(
                chirp_idx=chirp_idx,
                show_chirp=show_chirp,
                show_avg=show_avg,
                show_max_hold=show_max_hold,
            )
        )
        if len(traces) > 1:
            ax.legend(loc="upper right", ncol=min(len(traces), 4), fontsize=8)
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
        """Plot a 2D range-Doppler heatmap in dB (matplotlib imshow)."""
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
        """Overlay CFAR detection markers on a range-Doppler map."""
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
        """Plot azimuth spectrum in dB at a fixed range/Doppler bin."""
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
        """Attach config and a ``RadarProcessor`` for FFT-based Plotly charts."""
        self.config = config
        self.processor = RadarProcessor(config)

    def raw_adc_chirp(
        self,
        cube: np.ndarray,
        *,
        chirp_idx: int = 0,
        rx_idx: int | None = None,
    ) -> "go.Figure":
        """
        Interactive raw ADC time series for one or all RX channels.

        Legend entries are toggleable in the Plotly UI.
        """
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
        rx_idx: int | None = None,
        show_chirp: bool = True,
        show_avg: bool = False,
        show_max_hold: bool = False,
    ) -> "go.Figure":
        """
        Interactive range FFT profile in dB for each RX channel.

        Set ``show_chirp``, ``show_avg``, and/or ``show_max_hold`` to overlay
        single-chirp, averaged, and max-hold profiles. Legend entries are toggleable.
        """
        _require_plotly()
        r_cube = self.processor.range_fft(cube)
        range_axis = self.processor.compute_range_axis(r_cube.shape[0])
        rx_indices = _resolve_rx_indices(cube, all_rx=all_rx, rx_idx=rx_idx)
        traces = _iter_range_fft_profiles(
            r_cube,
            rx_indices=rx_indices,
            chirp_idx=chirp_idx,
            show_chirp=show_chirp,
            show_avg=show_avg,
            show_max_hold=show_max_hold,
        )
        fig = go.Figure()
        for label, profile, linestyle in traces:
            dash = _PLOTLY_DASH[linestyle]
            line = dict(width=1.2 if linestyle == "solid" else 1.4)
            if dash is not None:
                line["dash"] = dash
            fig.add_trace(
                go.Scatter(
                    x=range_axis,
                    y=20 * np.log10(profile + 1e-12),
                    mode="lines",
                    name=label,
                    line=line,
                )
            )
        fig.update_layout(
            title=_range_profile_title(
                chirp_idx=chirp_idx,
                show_chirp=show_chirp,
                show_avg=show_avg,
                show_max_hold=show_max_hold,
            ),
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
        """Interactive range-Doppler heatmap in dB."""
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
        """Range-Doppler heatmap with CFAR detection markers overlaid."""
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
        """Interactive azimuth spectrum in dB at a fixed range/Doppler bin."""
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
        """Bar chart of relative RX power from a QA report (0 dB = strongest)."""
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
        """Line plot of chirp RMS vs chirp index for each RX channel."""
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
        """Beat-frequency (range FFT) spectrum in dB for QA inspection."""
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
        """Bar chart of peak normalized xcorr vs a reference RX, with optional limit line."""
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
        """Heatmap of pairwise peak normalized RX cross-correlation."""
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
        """Display a Plotly figure once in the active notebook frontend."""
        fig.show()


# --- Backward-compatible functions ---

def plot_raw_adc_chirp(cube: np.ndarray, config: RadarConfig, **kwargs) -> plt.Axes:
    """Module-level wrapper for ``MatplotlibRadarPlotter.plot_raw_adc_chirp``."""
    return MatplotlibRadarPlotter(config).plot_raw_adc_chirp(cube, **kwargs)


def plot_range_profile(cube: np.ndarray, config: RadarConfig, **kwargs) -> plt.Axes:
    """Module-level wrapper for ``MatplotlibRadarPlotter.plot_range_profile``."""
    return MatplotlibRadarPlotter(config).plot_range_profile(cube, **kwargs)


def plot_range_doppler(rd_map, range_axis, doppler_axis, **kwargs) -> plt.Axes:
    """Module-level wrapper for ``MatplotlibRadarPlotter.plot_range_doppler``."""
    return MatplotlibRadarPlotter.plot_range_doppler(rd_map, range_axis, doppler_axis, **kwargs)


def plot_detections(rd_map, mask, range_axis, doppler_axis, **kwargs) -> plt.Axes:
    """Module-level wrapper: range-Doppler map with CFAR detections (matplotlib)."""
    ax = plot_range_doppler(rd_map, range_axis, doppler_axis, title="CFAR Detections", **kwargs)
    ri, di = np.where(mask)
    ax.scatter(doppler_axis[di], range_axis[ri], c="red", s=20, marker="x", label="Detections")
    ax.legend()
    return ax


def plot_angle_spectrum(rd_cube, config, range_idx, doppler_idx, **kwargs) -> plt.Axes:
    """Module-level wrapper for ``MatplotlibRadarPlotter.plot_angle_spectrum``."""
    return MatplotlibRadarPlotter(config).plot_angle_spectrum(
        rd_cube, range_idx, doppler_idx, **kwargs
    )


def iplot_raw_adc_chirp(cube: np.ndarray, config: RadarConfig, **kwargs) -> "go.Figure":
    """Module-level wrapper for ``PlotlyRadarPlotter.raw_adc_chirp``."""
    return PlotlyRadarPlotter(config).raw_adc_chirp(cube, **kwargs)


def iplot_range_profile(cube: np.ndarray, config: RadarConfig, **kwargs) -> "go.Figure":
    """Module-level wrapper for ``PlotlyRadarPlotter.range_profile``."""
    return PlotlyRadarPlotter(config).range_profile(cube, **kwargs)


def iplot_range_doppler(rd_map, range_axis, doppler_axis, **kwargs) -> "go.Figure":
    """Module-level wrapper for ``PlotlyRadarPlotter.range_doppler``."""
    return PlotlyRadarPlotter.range_doppler(rd_map, range_axis, doppler_axis, **kwargs)


def iplot_detections(rd_map, mask, range_axis, doppler_axis) -> "go.Figure":
    """Module-level wrapper for ``PlotlyRadarPlotter.detections``."""
    return PlotlyRadarPlotter.detections(rd_map, mask, range_axis, doppler_axis)


def iplot_angle_spectrum(rd_cube, config, range_idx, doppler_idx) -> "go.Figure":
    """Module-level wrapper for ``PlotlyRadarPlotter.angle_spectrum``."""
    return PlotlyRadarPlotter(config).angle_spectrum(rd_cube, range_idx, doppler_idx)


def iplot_qa_rx_power(report) -> "go.Figure":
    """Module-level wrapper for ``PlotlyRadarPlotter.qa_rx_power``."""
    return PlotlyRadarPlotter.qa_rx_power(report)


def iplot_qa_chirp_rms(report) -> "go.Figure":
    """Module-level wrapper for ``PlotlyRadarPlotter.qa_chirp_rms``."""
    return PlotlyRadarPlotter.qa_chirp_rms(report)


def iplot_qa_adc_spectrum(cube, config, **kwargs) -> "go.Figure":
    """Module-level wrapper for ``PlotlyRadarPlotter.qa_adc_spectrum``."""
    return PlotlyRadarPlotter(config).qa_adc_spectrum(cube, **kwargs)


def iplot_qa_rx_correlation(report, **kwargs) -> "go.Figure":
    """Module-level wrapper for ``PlotlyRadarPlotter.qa_rx_correlation``."""
    return PlotlyRadarPlotter.qa_rx_correlation(report, **kwargs)


def iplot_qa_rx_correlation_matrix(report) -> "go.Figure":
    """Module-level wrapper for ``PlotlyRadarPlotter.qa_rx_correlation_matrix``."""
    return PlotlyRadarPlotter.qa_rx_correlation_matrix(report)


show = PlotlyRadarPlotter.show
ishow = show
