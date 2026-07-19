"""Shared live radar frame container with lazy-processed products."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import time

import numpy as np

from ..devices import RadarConfig
from ..process import RadarProcessor
from ..testing import QAReport, QAThresholds, RawADCQA


@dataclass
class RadarFrame:
    """
    One live capture frame and cached derived products.

    Tabs should read from this object instead of re-running FFTs when possible.
    """

    cube: np.ndarray
    config: RadarConfig
    frame_id: int = 0
    timestamp: float = field(default_factory=time)
    source_name: str = "unknown"
    temperature_c: float | None = None
    qa_thresholds: QAThresholds = field(default_factory=QAThresholds)

    _processor: RadarProcessor | None = field(default=None, repr=False, compare=False)
    _range_cube: np.ndarray | None = field(default=None, repr=False, compare=False)
    _rd_map: np.ndarray | None = field(default=None, repr=False, compare=False)
    _rd_cube: np.ndarray | None = field(default=None, repr=False, compare=False)
    _qa_report: QAReport | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        self._processor = RadarProcessor(self.config)

    @property
    def processor(self) -> RadarProcessor:
        assert self._processor is not None
        return self._processor

    @property
    def range_cube(self) -> np.ndarray:
        """Complex range FFT cube ``(range_bins, chirps, rx)``."""
        if self._range_cube is None:
            self._range_cube = self.processor.range_fft(self.cube)
        return self._range_cube

    @property
    def rd_map(self) -> np.ndarray:
        """Non-coherent range-Doppler magnitude map."""
        if self._rd_map is None:
            self._rd_map, self._rd_cube = self.processor.process_range_doppler(self.cube)
        return self._rd_map

    @property
    def rd_cube(self) -> np.ndarray:
        """Complex range-Doppler cube retaining RX phase."""
        if self._rd_cube is None:
            self._rd_map, self._rd_cube = self.processor.process_range_doppler(self.cube)
        return self._rd_cube

    @property
    def qa_report(self) -> QAReport:
        """Raw ADC QA report for this frame (computed once)."""
        if self._qa_report is None:
            self._qa_report = RawADCQA(self.config, thresholds=self.qa_thresholds).run(
                self.cube, chirp_idx=0
            )
        return self._qa_report

    def range_axis(self) -> np.ndarray:
        return self.processor.compute_range_axis(self.range_cube.shape[0])

    def beat_freq_axis_mhz(self) -> np.ndarray:
        """Beat-frequency axis in MHz for the current range FFT size."""
        return self.processor.compute_beat_freq_axis(
            self.range_cube.shape[0], unit="mhz"
        )

    def doppler_axis(self) -> np.ndarray:
        return self.processor.compute_doppler_axis(self.rd_map.shape[1])

    def rd_map_for_rx(self, rx_indices: list[int] | None = None) -> np.ndarray:
        """
        Non-coherent range-Doppler map using only the given RX channels.

        If ``rx_indices`` is empty, returns a zero map with the same shape.
        If ``None`` or all RX, returns the cached full ``rd_map``.
        """
        rd = self.rd_cube
        n_rx = rd.shape[2]
        if rx_indices is None or len(rx_indices) == n_rx:
            return self.rd_map
        if not rx_indices:
            return np.zeros(rd.shape[:2], dtype=np.float64)
        return np.abs(rd[:, :, rx_indices]).sum(axis=2)

    def angle_axis(self, n_bins: int | None = None) -> np.ndarray:
        n = n_bins or (self.config.angle_fft_size or self.config.num_rx)
        return self.processor.compute_angle_axis(n)
