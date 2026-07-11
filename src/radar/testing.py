"""Generic FMCW raw ADC quality checks and reporting."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import RadarConfig

_INT16_FS = 32767.0


@dataclass
class QAThresholds:
    max_dc_offset_ratio: float = 0.35
    max_clip_fraction: float = 0.005
    max_rx_imbalance_db: float = 6.0
    max_chirp_rms_cv: float = 0.30
    min_chirp_rms: float = 1e-4
    chirp_duration_tol: float = 0.10
    min_rx_correlation: float = 0.50


@dataclass
class QACheck:
    name: str
    passed: bool
    detail: str


@dataclass
class RXStats:
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
    checks: list[QACheck] = field(default_factory=list)
    rx_stats: list[RXStats] = field(default_factory=list)
    chirp_rms: np.ndarray = field(default_factory=lambda: np.array([]))
    chirp_rms_per_rx: np.ndarray = field(default_factory=lambda: np.array([]))
    rx_power_db: np.ndarray = field(default_factory=lambda: np.array([]))
    rx_correlation: np.ndarray = field(default_factory=lambda: np.array([]))
    rx_correlation_matrix: np.ndarray = field(default_factory=lambda: np.array([]))

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    def summary(self) -> str:
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
        self.config = config
        self.thresholds = thresholds or QAThresholds()

    def run(
        self,
        cube: np.ndarray,
        *,
        raw_path: str | Path | None = None,
        chirp_idx: int = 0,
    ) -> QAReport:
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
        report.rx_correlation_matrix = self.compute_rx_correlation_matrix(
            cube, chirp_idx=chirp_idx
        )
        report.rx_correlation = report.rx_correlation_matrix[:, 0]

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
        config = self.config
        if config.data_dtype in ("int16", "uint16"):
            return _INT16_FS
        if config.data_dtype == "int32":
            return 2**31 - 1
        return 1.0

    def compute_rx_stats(self, cube: np.ndarray, *, chirp_idx: int = 0) -> list[RXStats]:
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
        return np.sqrt(np.mean(cube.astype(np.float64) ** 2, axis=(0, 2)))

    @staticmethod
    def compute_chirp_rms_per_rx(cube: np.ndarray) -> np.ndarray:
        return np.sqrt(np.mean(cube.astype(np.float64) ** 2, axis=0))

    @staticmethod
    def compute_rx_power_db(cube: np.ndarray, *, chirp_idx: int = 0) -> np.ndarray:
        power = np.mean(cube[:, chirp_idx, :].astype(np.float64) ** 2, axis=0)
        power = np.maximum(power, 1e-20)
        return 10.0 * np.log10(power / power.max())

    @staticmethod
    def compute_rx_correlation_matrix(
        cube: np.ndarray,
        *,
        chirp_idx: int | None = None,
    ) -> np.ndarray:
        num_rx = cube.shape[2]
        if chirp_idx is not None:
            data = np.asarray(cube[:, chirp_idx, :], dtype=np.float64)
        else:
            data = np.asarray(cube, dtype=np.float64).reshape(-1, num_rx)
        corr = np.eye(num_rx, dtype=np.float64)
        for i in range(num_rx):
            for j in range(i + 1, num_rx):
                r = RawADCQA._max_norm_xcorr(data[:, i], data[:, j])
                corr[i, j] = r
                corr[j, i] = r
        return corr

    @staticmethod
    def compute_rx_correlation(
        cube: np.ndarray,
        *,
        chirp_idx: int | None = None,
        ref_rx: int = 0,
    ) -> np.ndarray:
        matrix = RawADCQA.compute_rx_correlation_matrix(cube, chirp_idx=chirp_idx)
        ref = int(ref_rx)
        if ref < 0 or ref >= matrix.shape[0]:
            raise ValueError(f"ref_rx={ref_rx} out of range for {matrix.shape[0]} RX channels")
        return matrix[:, ref]

    @staticmethod
    def _max_norm_xcorr(a: np.ndarray, b: np.ndarray) -> float:
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        a = a - a.mean()
        b = b - b.mean()
        a_std = float(a.std())
        b_std = float(b.std())
        if a_std == 0.0 or b_std == 0.0:
            return 0.0
        a = a / a_std
        b = b / b_std
        xcorr = np.correlate(a, b, mode="full") / a.size
        return float(np.max(np.abs(xcorr)))

    def validate_file_size(self, path: str | Path) -> QACheck:
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
    return RawADCQA(config).compute_rx_stats(cube, **kwargs)


def compute_chirp_rms(cube: np.ndarray) -> np.ndarray:
    return RawADCQA.compute_chirp_rms(cube)


def compute_chirp_rms_per_rx(cube: np.ndarray) -> np.ndarray:
    return RawADCQA.compute_chirp_rms_per_rx(cube)


def compute_rx_power_db(cube: np.ndarray, **kwargs) -> np.ndarray:
    return RawADCQA.compute_rx_power_db(cube, **kwargs)


def compute_rx_correlation_matrix(cube: np.ndarray, **kwargs) -> np.ndarray:
    return RawADCQA.compute_rx_correlation_matrix(cube, **kwargs)


def compute_rx_correlation(cube: np.ndarray, **kwargs) -> np.ndarray:
    return RawADCQA.compute_rx_correlation(cube, **kwargs)


def validate_file_size(path: str | Path, config: RadarConfig) -> QACheck:
    return RawADCQA(config).validate_file_size(path)


def run_raw_adc_qa(
    cube: np.ndarray,
    config: RadarConfig,
    *,
    thresholds: QAThresholds | None = None,
    raw_path: str | Path | None = None,
    chirp_idx: int = 0,
) -> QAReport:
    return RawADCQA(config, thresholds=thresholds).run(
        cube, raw_path=raw_path, chirp_idx=chirp_idx
    )
