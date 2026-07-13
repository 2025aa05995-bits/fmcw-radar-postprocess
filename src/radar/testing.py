"""Generic FMCW raw ADC quality checks and reporting."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import RadarConfig

_INT16_FS = 32767.0


@dataclass
class QAThresholds:
    """Pass/fail limits used by ``RawADCQA`` when evaluating a capture."""

    max_dc_offset_ratio: float = 0.35
    max_clip_fraction: float = 0.005
    max_rx_imbalance_db: float = 6.0
    max_chirp_rms_cv: float = 0.30
    min_chirp_rms: float = 1e-4
    chirp_duration_tol: float = 0.10
    min_rx_correlation: float = 0.50


@dataclass
class QACheck:
    """Single QA check result with a name, pass flag, and human-readable detail."""

    name: str
    passed: bool
    detail: str


@dataclass
class RXStats:
    """Per-RX statistics for one chirp slice of the ADC cube."""

    rx_idx: int
    mean: float
    std: float
    rms: float
    min_val: float
    max_val: float
    clip_fraction: float
    dc_offset_ratio: float


@dataclass
class QAReport:
    """
    Aggregated raw ADC QA output.

    Contains individual checks, per-RX stats, chirp stability metrics,
    RX correlation arrays, and RX group-delay estimates from ``RawADCQA.run``.
    """

    checks: list[QACheck] = field(default_factory=list)
    rx_stats: list[RXStats] = field(default_factory=list)
    chirp_rms: np.ndarray = field(default_factory=lambda: np.array([]))
    chirp_rms_per_rx: np.ndarray = field(default_factory=lambda: np.array([]))
    rx_power_db: np.ndarray = field(default_factory=lambda: np.array([]))
    rx_correlation: np.ndarray = field(default_factory=lambda: np.array([]))
    rx_correlation_matrix: np.ndarray = field(default_factory=lambda: np.array([]))
    rx_group_delay_matrix: np.ndarray = field(default_factory=lambda: np.array([]))
    rx_group_delay_samples: np.ndarray = field(default_factory=lambda: np.array([]))

    @property
    def passed(self) -> bool:
        """True when every check in ``checks`` passed."""
        return all(c.passed for c in self.checks)

    def summary(self) -> str:
        """Return a multi-line text summary of all checks and overall pass/fail."""
        lines = ["Raw ADC QA summary", "=" * 40]
        for chk in self.checks:
            mark = "PASS" if chk.passed else "FAIL"
            lines.append(f"[{mark}] {chk.name}: {chk.detail}")
        lines.append("-" * 40)
        lines.append(f"Overall: {'PASS' if self.passed else 'FAIL'}")
        return "\n".join(lines)


class RawADCQA:
    """Generic FMCW raw ADC quality analysis."""

    def __init__(
        self,
        config: RadarConfig,
        *,
        thresholds: QAThresholds | None = None,
    ) -> None:
        """Attach radar config and optional custom QA thresholds."""
        self.config = config
        self.thresholds = thresholds or QAThresholds()

    def run(
        self,
        cube: np.ndarray,
        *,
        raw_path: str | Path | None = None,
        chirp_idx: int = 0,
    ) -> QAReport:
        """
        Run the full raw ADC QA suite on ``cube``.

        Validates shape, timing, data type, finiteness, per-RX DC/clipping,
        RX balance, correlation, and chirp RMS stability. Optionally checks
        ``raw_path`` file size against expected byte count.
        """
        th = self.thresholds
        config = self.config
        report = QAReport()

        expected = (config.num_samples, config.num_chirps, config.num_rx)
        ok = cube.shape == expected
        report.checks.append(
            QACheck("cube_shape", ok, f"expected {expected}, got {cube.shape}")
        )

        t_adc = config.num_samples / config.adc_sample_rate_hz
        t_ramp = config.chirp_duration_s
        rel_err = abs(t_adc - t_ramp) / t_ramp if t_ramp > 0 else 0.0
        report.checks.append(
            QACheck(
                "chirp_timing",
                rel_err <= th.chirp_duration_tol,
                f"ADC window {t_adc*1e6:.2f} µs vs ramp {t_ramp*1e6:.2f} µs "
                f"(error {100*rel_err:.1f}%, limit {100*th.chirp_duration_tol:.0f}%)",
            )
        )

        if raw_path is not None:
            report.checks.append(self.validate_file_size(raw_path))

        if np.iscomplexobj(cube):
            report.checks.append(
                QACheck("data_type", False, "cube is complex — expected real ADC samples")
            )
        else:
            report.checks.append(QACheck("data_type", True, f"real {cube.dtype}"))

        if np.any(~np.isfinite(cube)):
            report.checks.append(QACheck("finite_samples", False, "NaN or Inf in cube"))
        else:
            report.checks.append(QACheck("finite_samples", True, "all samples finite"))

        report.rx_stats = self.compute_rx_stats(cube, chirp_idx=chirp_idx)
        report.rx_power_db = self.compute_rx_power_db(cube, chirp_idx=chirp_idx)
        report.chirp_rms = self.compute_chirp_rms(cube)
        report.chirp_rms_per_rx = self.compute_chirp_rms_per_rx(cube)
        (
            report.rx_correlation_matrix,
            report.rx_group_delay_matrix,
        ) = self.compute_rx_correlation_and_delay_matrices(cube, chirp_idx=chirp_idx)
        report.rx_correlation = report.rx_correlation_matrix[:, 0]
        report.rx_group_delay_samples = report.rx_group_delay_matrix[0, :]

        for st in report.rx_stats:
            report.checks.append(
                QACheck(
                    f"rx{st.rx_idx}_dc_offset",
                    st.dc_offset_ratio <= th.max_dc_offset_ratio,
                    f"|mean|/std = {st.dc_offset_ratio:.3f} (limit {th.max_dc_offset_ratio})",
                )
            )
            report.checks.append(
                QACheck(
                    f"rx{st.rx_idx}_clipping",
                    st.clip_fraction <= th.max_clip_fraction,
                    f"clip fraction = {100*st.clip_fraction:.3f}% "
                    f"(limit {100*th.max_clip_fraction:.1f}%)",
                )
            )

        spread_db = float(report.rx_power_db.max() - report.rx_power_db.min())
        report.checks.append(
            QACheck(
                "rx_balance",
                spread_db <= th.max_rx_imbalance_db,
                f"RX power spread = {spread_db:.1f} dB (limit {th.max_rx_imbalance_db:.1f} dB)",
            )
        )

        if report.rx_correlation.shape[0] >= 2:
            ref_rx = 0
            others = [i for i in range(report.rx_correlation.shape[0]) if i != ref_rx]
            min_corr = float(np.min(report.rx_correlation[others]))
            min_rx = int(others[int(np.argmin(report.rx_correlation[others]))])
            report.checks.append(
                QACheck(
                    "rx_correlation",
                    min_corr >= th.min_rx_correlation,
                    f"min xcorr vs RX{ref_rx}: RX{min_rx} = {min_corr:.3f} "
                    f"(limit {th.min_rx_correlation:.2f})",
                )
            )

        rms = report.chirp_rms
        rms_mean = float(rms.mean()) if rms.size else 0.0
        rms_cv = float(rms.std() / rms_mean) if rms_mean > 0 else 0.0
        report.checks.append(
            QACheck(
                "chirp_stability",
                rms_cv <= th.max_chirp_rms_cv and rms_mean >= th.min_chirp_rms,
                f"RMS mean={rms_mean:.4g}, CV={rms_cv:.3f} "
                f"(CV limit {th.max_chirp_rms_cv}, min RMS {th.min_chirp_rms})",
            )
        )
        return report

    def adc_full_scale(self) -> float:
        """Return the nominal full-scale ADC value for clipping detection."""
        config = self.config
        if config.data_dtype in ("int16", "uint16"):
            return _INT16_FS
        if config.data_dtype == "int32":
            return 2**31 - 1
        return 1.0

    def compute_rx_stats(self, cube: np.ndarray, *, chirp_idx: int = 0) -> list[RXStats]:
        """
        Compute mean, RMS, clipping fraction, and DC offset ratio per RX.

        Uses the fast-time slice at ``chirp_idx`` for each receiver channel.
        """
        fs = self.adc_full_scale()
        clip_level = 0.98 * fs
        stats: list[RXStats] = []
        for rx in range(cube.shape[2]):
            x = np.asarray(cube[:, chirp_idx, rx], dtype=np.float64)
            std = float(x.std())
            mean = float(x.mean())
            stats.append(
                RXStats(
                    rx_idx=rx,
                    mean=mean,
                    std=std,
                    rms=float(np.sqrt(np.mean(x**2))),
                    min_val=float(x.min()),
                    max_val=float(x.max()),
                    clip_fraction=float(np.mean(np.abs(x) >= clip_level)),
                    dc_offset_ratio=float(abs(mean) / std) if std > 0 else 0.0,
                )
            )
        return stats

    @staticmethod
    def compute_chirp_rms(cube: np.ndarray) -> np.ndarray:
        """Return RMS per chirp, averaged over samples and RX (shape ``num_chirps``)."""
        return np.sqrt(np.mean(cube.astype(np.float64) ** 2, axis=(0, 2)))

    @staticmethod
    def compute_chirp_rms_per_rx(cube: np.ndarray) -> np.ndarray:
        """Return RMS per (chirp, RX) with shape ``(num_chirps, num_rx)``."""
        return np.sqrt(np.mean(cube.astype(np.float64) ** 2, axis=0))

    @staticmethod
    def compute_rx_power_db(cube: np.ndarray, *, chirp_idx: int = 0) -> np.ndarray:
        """Return relative RX power in dB (0 dB = strongest channel) for one chirp."""
        power = np.mean(cube[:, chirp_idx, :].astype(np.float64) ** 2, axis=0)
        power = np.maximum(power, 1e-20)
        return 10.0 * np.log10(power / power.max())

    @staticmethod
    def _rx_qa_data(cube: np.ndarray, *, chirp_idx: int | None) -> np.ndarray:
        """Return 2D fast-time data with shape ``(num_samples_or_flattened, num_rx)``."""
        num_rx = cube.shape[2]
        if chirp_idx is not None:
            return np.asarray(cube[:, chirp_idx, :], dtype=np.float64)
        return np.asarray(cube, dtype=np.float64).reshape(-1, num_rx)

    @staticmethod
    def _peak_norm_xcorr(a: np.ndarray, b: np.ndarray) -> tuple[float, int]:
        """
        Peak normalized cross-correlation and lag between two 1D signals.

        Returns ``(peak_abs_corr, lag_samples)`` where positive lag means ``b``
        is delayed relative to ``a`` (shift ``b`` right by lag samples to align).
        """
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        a = a - a.mean()
        b = b - b.mean()
        a_std = float(a.std())
        b_std = float(b.std())
        if a_std == 0.0 or b_std == 0.0:
            return 0.0, 0
        a = a / a_std
        b = b / b_std
        xcorr = np.correlate(a, b, mode="full") / a.size
        peak_idx = int(np.argmax(np.abs(xcorr)))
        # Positive lag => ``b`` is delayed relative to ``a`` (appears later in time).
        lag = (b.size - 1) - peak_idx
        return float(np.abs(xcorr[peak_idx])), lag

    @staticmethod
    def compute_rx_correlation_and_delay_matrices(
        cube: np.ndarray,
        *,
        chirp_idx: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Build RX correlation and group-delay matrices from peak normalized xcorr.

        ``correlation[i, j]`` is in [0, 1]. ``delay[i, j]`` is the lag in ADC
        samples such that RX ``j`` best aligns to RX ``i``; positive means ``j``
        lags ``i``. The delay matrix is antisymmetric with a zero diagonal.
        """
        data = RawADCQA._rx_qa_data(cube, chirp_idx=chirp_idx)
        num_rx = data.shape[1]
        corr = np.eye(num_rx, dtype=np.float64)
        delay = np.zeros((num_rx, num_rx), dtype=np.float64)
        for i in range(num_rx):
            for j in range(i + 1, num_rx):
                peak, lag = RawADCQA._peak_norm_xcorr(data[:, i], data[:, j])
                corr[i, j] = peak
                corr[j, i] = peak
                delay[i, j] = float(lag)
                delay[j, i] = float(-lag)
        return corr, delay

    @staticmethod
    def compute_rx_correlation_matrix(
        cube: np.ndarray,
        *,
        chirp_idx: int | None = None,
    ) -> np.ndarray:
        """
        Build symmetric RX correlation matrix using peak normalized cross-correlation.

        Phase-steering between channels does not reduce the metric; values lie in [0, 1].
        """
        corr, _ = RawADCQA.compute_rx_correlation_and_delay_matrices(
            cube, chirp_idx=chirp_idx
        )
        return corr

    @staticmethod
    def compute_rx_group_delay_matrix(
        cube: np.ndarray,
        *,
        chirp_idx: int | None = None,
    ) -> np.ndarray:
        """
        Build antisymmetric RX group-delay matrix in ADC samples.

        ``delay[i, j]`` is positive when RX ``j`` lags RX ``i`` at peak xcorr.
        """
        _, delay = RawADCQA.compute_rx_correlation_and_delay_matrices(
            cube, chirp_idx=chirp_idx
        )
        return delay

    @staticmethod
    def compute_rx_group_delay_samples(
        cube: np.ndarray,
        *,
        chirp_idx: int | None = None,
        ref_rx: int = 0,
    ) -> np.ndarray:
        """Return group delay of each RX relative to ``ref_rx`` (in ADC samples)."""
        delay = RawADCQA.compute_rx_group_delay_matrix(cube, chirp_idx=chirp_idx)
        ref = int(ref_rx)
        if ref < 0 or ref >= delay.shape[0]:
            raise ValueError(f"ref_rx={ref_rx} out of range for {delay.shape[0]} RX channels")
        return delay[ref, :]

    @staticmethod
    def compute_rx_correlation(
        cube: np.ndarray,
        *,
        chirp_idx: int | None = None,
        ref_rx: int = 0,
    ) -> np.ndarray:
        """Return one column of the correlation matrix vs ``ref_rx``."""
        matrix = RawADCQA.compute_rx_correlation_matrix(cube, chirp_idx=chirp_idx)
        ref = int(ref_rx)
        if ref < 0 or ref >= matrix.shape[0]:
            raise ValueError(f"ref_rx={ref_rx} out of range for {matrix.shape[0]} RX channels")
        return matrix[:, ref]

    @staticmethod
    def _max_norm_xcorr(a: np.ndarray, b: np.ndarray) -> float:
        """Peak absolute value of normalized cross-correlation between two 1D signals."""
        peak, _ = RawADCQA._peak_norm_xcorr(a, b)
        return peak

    def validate_file_size(self, path: str | Path) -> QACheck:
        """Check that a raw capture file is at least as large as expected from config."""
        path = Path(path)
        config = self.config
        if not path.exists():
            return QACheck("file_size", False, f"File not found: {path}")
        dtype_bytes = {"int16": 2, "uint16": 2, "int32": 4, "float32": 4, "float64": 8}.get(
            config.data_dtype, 2
        )
        expected = config.num_samples * config.num_chirps * config.num_rx * dtype_bytes
        actual = path.stat().st_size
        ok = actual >= expected
        return QACheck(
            "file_size",
            ok,
            f"expected >={expected} bytes, got {actual} bytes",
        )


def compute_rx_stats(cube: np.ndarray, config: RadarConfig, **kwargs) -> list[RXStats]:
    """Module-level wrapper for ``RawADCQA.compute_rx_stats``."""
    return RawADCQA(config).compute_rx_stats(cube, **kwargs)


def compute_chirp_rms(cube: np.ndarray) -> np.ndarray:
    """Module-level wrapper for ``RawADCQA.compute_chirp_rms``."""
    return RawADCQA.compute_chirp_rms(cube)


def compute_chirp_rms_per_rx(cube: np.ndarray) -> np.ndarray:
    """Module-level wrapper for ``RawADCQA.compute_chirp_rms_per_rx``."""
    return RawADCQA.compute_chirp_rms_per_rx(cube)


def compute_rx_power_db(cube: np.ndarray, **kwargs) -> np.ndarray:
    """Module-level wrapper for ``RawADCQA.compute_rx_power_db``."""
    return RawADCQA.compute_rx_power_db(cube, **kwargs)


def compute_rx_correlation_matrix(cube: np.ndarray, **kwargs) -> np.ndarray:
    """Module-level wrapper for ``RawADCQA.compute_rx_correlation_matrix``."""
    return RawADCQA.compute_rx_correlation_matrix(cube, **kwargs)


def compute_rx_group_delay_matrix(cube: np.ndarray, **kwargs) -> np.ndarray:
    """Module-level wrapper for ``RawADCQA.compute_rx_group_delay_matrix``."""
    return RawADCQA.compute_rx_group_delay_matrix(cube, **kwargs)


def compute_rx_group_delay_samples(cube: np.ndarray, **kwargs) -> np.ndarray:
    """Module-level wrapper for ``RawADCQA.compute_rx_group_delay_samples``."""
    return RawADCQA.compute_rx_group_delay_samples(cube, **kwargs)


def compute_rx_correlation(cube: np.ndarray, **kwargs) -> np.ndarray:
    """Module-level wrapper for ``RawADCQA.compute_rx_correlation``."""
    return RawADCQA.compute_rx_correlation(cube, **kwargs)


def validate_file_size(path: str | Path, config: RadarConfig) -> QACheck:
    """Module-level wrapper for ``RawADCQA.validate_file_size``."""
    return RawADCQA(config).validate_file_size(path)


def run_raw_adc_qa(
    cube: np.ndarray,
    config: RadarConfig,
    *,
    thresholds: QAThresholds | None = None,
    raw_path: str | Path | None = None,
    chirp_idx: int = 0,
) -> QAReport:
    """Module-level wrapper for ``RawADCQA.run``."""
    return RawADCQA(config, thresholds=thresholds).run(
        cube, raw_path=raw_path, chirp_idx=chirp_idx
    )
