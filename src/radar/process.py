"""FMCW radar signal processing and pipeline orchestration."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.signal import windows

from .config import (
    SPEED_OF_LIGHT,
    RadarConfig,
    RadarDataCube,
    RadarDataIO,
    generate_synthetic_cube,
    load_config,
    load_radar_cube,
)
from .testing import QAReport, QAThresholds, RawADCQA


def _next_pow2(n: int) -> int:
    """Return the smallest power of two >= n."""
    return 1 << (n - 1).bit_length()


def _window(n: int, name: str = "hann") -> np.ndarray:
    """Return a 1D window of length n (hann, hamming, blackman, or ones)."""
    if name == "hann":
        return windows.hann(n, sym=False)
    if name == "hamming":
        return windows.hamming(n, sym=False)
    if name == "blackman":
        return windows.blackman(n, sym=False)
    return np.ones(n)


def _range_n_fft(config: RadarConfig, n_fft: int | None) -> int:
    """Resolve range FFT size from explicit value, config, or next power of two."""
    return n_fft or config.range_fft_size or _next_pow2(config.num_samples)


class RadarProcessor:
    """
    Range / Doppler / Angle FFT pipeline and CFAR detection.

    Operates on real ADC cubes with shape ``(samples, chirps, rx)``.
    Real input uses ``rfft`` on the fast-time axis to avoid mirrored range bins.
    """

    def __init__(self, config: RadarConfig) -> None:
        """Attach a ``RadarConfig`` used for FFT sizes and axis computation."""
        self.config = config

    def range_fft(
        self,
        cube: np.ndarray,
        *,
        n_fft: int | None = None,
        window: str = "hann",
    ) -> np.ndarray:
        """
        1D range FFT along the fast-time (samples) axis.

        Applies a window, then ``rfft`` for real input. Returns complex cube
        with shape ``(range_bins, chirps, rx)``.
        """
        config = self.config
        n_fft = _range_n_fft(config, n_fft)
        win = _window(config.num_samples, window)
        windowed = cube * win[:, None, None]
        if np.isrealobj(windowed) and not np.iscomplexobj(windowed):
            return np.fft.rfft(windowed, n=n_fft, axis=0)
        return np.fft.fft(windowed, n=n_fft, axis=0)[: n_fft // 2 + 1]

    def doppler_fft(
        self,
        range_cube: np.ndarray,
        *,
        n_fft: int | None = None,
        window: str = "hann",
    ) -> np.ndarray:
        """
        2D Doppler FFT along the chirp axis.

        Returns complex cube with shape ``(range_bins, doppler_bins, rx)``,
        Doppler axis fftshifted to centre zero velocity.
        """
        config = self.config
        n_fft = n_fft or config.doppler_fft_size or _next_pow2(config.num_chirps)
        win = _window(range_cube.shape[1], window)
        windowed = range_cube * win[None, :, None]
        rd = np.fft.fft(windowed, n=n_fft, axis=1)
        return np.fft.fftshift(rd, axes=1)

    def angle_fft(
        self,
        rd_cube: np.ndarray,
        *,
        n_fft: int | None = None,
    ) -> np.ndarray:
        """
        3D angle (azimuth) FFT along the RX channel axis.

        Returns complex cube with shape ``(range_bins, doppler_bins, angle_bins)``,
        angle axis fftshifted.
        """
        config = self.config
        n_fft = n_fft or config.angle_fft_size or _next_pow2(config.num_rx)
        rd_angle = np.fft.fft(rd_cube, n=n_fft, axis=2)
        return np.fft.fftshift(rd_angle, axes=2)

    def process_range_doppler(
        self,
        cube: np.ndarray,
        *,
        combine_rx: str = "sum",
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Run range then Doppler FFT and produce a 2D magnitude map.

        Returns ``(rd_map, rd_cube)`` where ``rd_map`` is non-coherent RX
        combination (sum, max, or first RX) and ``rd_cube`` retains phase.
        """
        r_cube = self.range_fft(cube)
        rd_cube = self.doppler_fft(r_cube)
        if combine_rx == "sum":
            rd_map = np.abs(rd_cube).sum(axis=2)
        elif combine_rx == "max":
            rd_map = np.abs(rd_cube).max(axis=2)
        else:
            rd_map = np.abs(rd_cube[:, :, 0])
        return rd_map, rd_cube

    @staticmethod
    def cfar_2d(
        rd_map: np.ndarray,
        *,
        guard_cells: tuple[int, int] = (2, 2),
        train_cells: tuple[int, int] = (8, 8),
        pfa: float = 1e-4,
        threshold_db: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        2D cell-averaging CFAR on a range-Doppler magnitude map.

        For each cell, estimates noise from surrounding training cells
        (excluding guard region) and flags detections above threshold.
        Returns ``(detection_mask, threshold_map)``.
        """
        guard_r, guard_d = guard_cells
        train_r, train_d = train_cells
        nr, nd = rd_map.shape
        threshold = np.full_like(rd_map, np.inf, dtype=np.float64)
        mask = np.zeros(rd_map.shape, dtype=bool)
        n_train = (2 * train_r + 2 * guard_r + 1) * (2 * train_d + 2 * guard_d + 1)
        n_train -= (2 * guard_r + 1) * (2 * guard_d + 1)
        alpha = n_train * (pfa ** (-1.0 / n_train) - 1) if threshold_db is None else None

        for ri in range(train_r + guard_r, nr - train_r - guard_r):
            for di in range(train_d + guard_d, nd - train_d - guard_d):
                r0, r1 = ri - train_r - guard_r, ri + train_r + guard_r + 1
                d0, d1 = di - train_d - guard_d, di + train_d + guard_d + 1
                window = rd_map[r0:r1, d0:d1].astype(np.float64)
                gr0, gd0 = train_r, train_d
                gr1, gd1 = window.shape[0] - train_r, window.shape[1] - train_d
                window[gr0:gr1, gd0:gd1] = 0
                noise = window.sum() / max(n_train, 1)
                thresh = noise * (10 ** (threshold_db / 10)) if threshold_db is not None else alpha * noise
                threshold[ri, di] = thresh
                mask[ri, di] = rd_map[ri, di] > thresh
        return mask, threshold

    def compute_range_axis(self, n_bins: int, *, n_fft: int | None = None) -> np.ndarray:
        """
        Convert range FFT bin indices to metres.

        Uses beat frequency from ``rfft`` bins and chirp slope; valid for
        positive beat frequencies only.
        """
        config = self.config
        n_fft = n_fft or _range_n_fft(config, None)
        freq_res = config.adc_sample_rate_hz / n_fft
        freqs = np.arange(n_bins) * freq_res
        return freqs * SPEED_OF_LIGHT / (2.0 * config.chirp_slope)

    def compute_doppler_axis(self, n_bins: int) -> np.ndarray:
        """Convert Doppler FFT bin indices to velocity in m/s (centred on zero)."""
        config = self.config
        doppler_freq = np.fft.fftshift(np.fft.fftfreq(n_bins, d=config.chirp_period_s))
        return doppler_freq * config.wavelength_m / 2.0

    def compute_angle_axis(self, n_bins: int) -> np.ndarray:
        """
        Convert angle FFT bin indices to azimuth in degrees.

        Uses ULA narrow-angle approximation: sin(θ) ≈ u × λ / d.
        """
        config = self.config
        u = np.fft.fftshift(np.fft.fftfreq(n_bins, d=config.rx_spacing_m))
        u = np.clip(u * config.wavelength_m, -1.0, 1.0)
        return np.rad2deg(np.arcsin(u))


class FMCWRadarPipeline:
    """
    High-level OOP session for FMCW offline post-processing.

    Bundles config, data cube, processor, QA, and plotters. Caches intermediate
    results (``rd_map``, ``rd_cube``, ``det_mask``, ``qa_report``) on the instance.
    """

    def __init__(
        self,
        config: RadarConfig,
        *,
        qa_thresholds: QAThresholds | None = None,
    ) -> None:
        """Create a pipeline session with the given radar configuration."""
        self.config = config
        self.cube: np.ndarray | None = None
        self.processor = RadarProcessor(config)
        self.qa = RawADCQA(config, thresholds=qa_thresholds)
        self.qa_report: QAReport | None = None
        self.rd_map: np.ndarray | None = None
        self.rd_cube: np.ndarray | None = None
        self.det_mask: np.ndarray | None = None
        self.plotly = None
        self.matplotlib = None
        self._init_plotters()

    @classmethod
    def from_config(cls, path: str | Path, **kwargs) -> FMCWRadarPipeline:
        """Build a pipeline from an XML configuration file path."""
        return cls(RadarConfig.from_xml(path), **kwargs)

    def load_synthetic(
        self,
        *,
        targets: list[tuple[float, float, float]] | None = None,
        snr_db: float = 25.0,
        seed: int = 42,
    ) -> np.ndarray:
        """
        Generate and store a synthetic cube on this pipeline instance.

        Returns the cube array; also sets ``self.cube``.
        """
        self.cube = RadarDataIO.synthetic(
            self.config, targets=targets, snr_db=snr_db, seed=seed
        )
        return self.cube

    def load_cube(
        self,
        path: str | Path,
        *,
        offset_bytes: int = 0,
        scale: float = 1.0,
    ) -> np.ndarray:
        """
        Load a raw capture from disk and store it on this pipeline instance.

        Returns the cube array; also sets ``self.cube``.
        """
        self.cube = RadarDataIO.load(
            path, self.config, offset_bytes=offset_bytes, scale=scale
        )
        return self.cube

    def load_data_cube(self, data_cube: RadarDataCube) -> np.ndarray:
        """
        Adopt a ``RadarDataCube`` wrapper; refreshes services if config changed.

        Returns the cube array; also sets ``self.cube``.
        """
        if data_cube.config is not self.config:
            self.config = data_cube.config
            self._refresh_services()
        self.cube = data_cube.data
        return self.cube

    def _init_plotters(self) -> None:
        """Lazily construct matplotlib and Plotly plotter instances."""
        from .visualization import MatplotlibRadarPlotter, PlotlyRadarPlotter

        self.plotly = PlotlyRadarPlotter(self.config)
        self.matplotlib = MatplotlibRadarPlotter(self.config)

    def _refresh_services(self) -> None:
        """Rebuild processor, QA, and plotters after a config change."""
        th = self.qa.thresholds
        self.processor = RadarProcessor(self.config)
        self.qa = RawADCQA(self.config, thresholds=th)
        self._init_plotters()

    def run_qa(
        self,
        *,
        raw_path: str | Path | None = None,
        chirp_idx: int = 0,
    ) -> QAReport:
        """
        Run raw ADC QA checks on the loaded cube.

        Optionally validates ``raw_path`` file size. Stores result in
        ``self.qa_report`` and returns it.
        """
        if self.cube is None:
            raise RuntimeError("No cube loaded — call load_synthetic() or load_cube() first")
        self.qa_report = self.qa.run(self.cube, raw_path=raw_path, chirp_idx=chirp_idx)
        return self.qa_report

    def process_range_doppler(self, *, combine_rx: str = "sum") -> tuple[np.ndarray, np.ndarray]:
        """
        Process the loaded cube to a range-Doppler map.

        Stores ``self.rd_map`` and ``self.rd_cube`` and returns them.
        """
        if self.cube is None:
            raise RuntimeError("No cube loaded — call load_synthetic() or load_cube() first")
        self.rd_map, self.rd_cube = self.processor.process_range_doppler(
            self.cube, combine_rx=combine_rx
        )
        return self.rd_map, self.rd_cube

    def cfar(
        self,
        rd_map: np.ndarray | None = None,
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Run 2D CFAR on a range-Doppler map (defaults to ``self.rd_map``).

        Stores ``self.det_mask`` and returns ``(det_mask, threshold_map)``.
        """
        rd_map = rd_map if rd_map is not None else self.rd_map
        if rd_map is None:
            raise RuntimeError("No range-Doppler map — call process_range_doppler() first")
        self.det_mask, threshold = RadarProcessor.cfar_2d(rd_map, **kwargs)
        return self.det_mask, threshold

    def range_axis(self, n_bins: int | None = None) -> np.ndarray:
        """
        Return the range axis in metres for the current or inferred bin count.

        Infers ``n_bins`` from ``self.rd_map`` or a range FFT of ``self.cube``.
        """
        if n_bins is None:
            if self.rd_map is not None:
                n_bins = self.rd_map.shape[0]
            elif self.cube is not None:
                n_bins = self.processor.range_fft(self.cube).shape[0]
            else:
                raise RuntimeError("No data available for range axis")
        return self.processor.compute_range_axis(n_bins)

    def doppler_axis(self, n_bins: int | None = None) -> np.ndarray:
        """
        Return the Doppler velocity axis in m/s for the current bin count.

        Infers ``n_bins`` from ``self.rd_map`` when not provided.
        """
        if n_bins is None:
            if self.rd_map is not None:
                n_bins = self.rd_map.shape[1]
            else:
                raise RuntimeError("No range-Doppler map for doppler axis")
        return self.processor.compute_doppler_axis(n_bins)

    def show(self, fig) -> None:
        """Display a Plotly figure once in the active Jupyter/notebook frontend."""
        self.plotly.show(fig)


# --- Backward-compatible module-level wrappers ---

def range_fft(cube: np.ndarray, config: RadarConfig, **kwargs) -> np.ndarray:
    """Module-level wrapper for ``RadarProcessor.range_fft``."""
    return RadarProcessor(config).range_fft(cube, **kwargs)


def doppler_fft(range_cube: np.ndarray, config: RadarConfig, **kwargs) -> np.ndarray:
    """Module-level wrapper for ``RadarProcessor.doppler_fft``."""
    return RadarProcessor(config).doppler_fft(range_cube, **kwargs)


def angle_fft(rd_cube: np.ndarray, config: RadarConfig, **kwargs) -> np.ndarray:
    """Module-level wrapper for ``RadarProcessor.angle_fft``."""
    return RadarProcessor(config).angle_fft(rd_cube, **kwargs)


def process_range_doppler(cube: np.ndarray, config: RadarConfig, **kwargs) -> tuple[np.ndarray, np.ndarray]:
    """Module-level wrapper for ``RadarProcessor.process_range_doppler``."""
    return RadarProcessor(config).process_range_doppler(cube, **kwargs)


def cfar_2d(rd_map: np.ndarray, **kwargs) -> tuple[np.ndarray, np.ndarray]:
    """Module-level wrapper for ``RadarProcessor.cfar_2d``."""
    return RadarProcessor.cfar_2d(rd_map, **kwargs)


def compute_range_axis(config: RadarConfig, n_bins: int, **kwargs) -> np.ndarray:
    """Module-level wrapper for ``RadarProcessor.compute_range_axis``."""
    return RadarProcessor(config).compute_range_axis(n_bins, **kwargs)


def compute_doppler_axis(config: RadarConfig, n_bins: int) -> np.ndarray:
    """Module-level wrapper for ``RadarProcessor.compute_doppler_axis``."""
    return RadarProcessor(config).compute_doppler_axis(n_bins)


def compute_angle_axis(config: RadarConfig, n_bins: int) -> np.ndarray:
    """Module-level wrapper for ``RadarProcessor.compute_angle_axis``."""
    return RadarProcessor(config).compute_angle_axis(n_bins)
