"""Synthetic radar device — software-generated live frames (no hardware)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .base import RadarConfig, RadarDevice, project_sample_config_path
from .factory import RadarDeviceFactory

# Reference levels for amplitude scaling (0 dB relative at these settings).
_TX_PWR_REF_DB = 0.0
_RX_GAIN_REF_DB = 26.0


def generate_synthetic_cube(
    config: RadarConfig,
    *,
    targets: list[tuple[float, float, float]] | None = None,
    snr_db: float = 25.0,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate a synthetic real-valued FMCW radar cube.

    Models beat-frequency cosines with array phase steering and additive
    Gaussian noise. Each target is ``(range_m, velocity_mps, angle_deg)``.
    """
    rng = np.random.default_rng(seed)
    ns, nc, nr = config.num_samples, config.num_chirps, config.num_rx
    if targets is None:
        targets = [(15.0, 3.0, 10.0), (40.0, -5.0, -20.0)]

    t_fast = np.arange(ns) / config.adc_sample_rate_hz
    t_slow = np.arange(nc) * config.chirp_period_s
    rx_pos = np.arange(nr) * config.rx_spacing_m
    cube = np.zeros((ns, nc, nr), dtype=np.float32)

    for range_m, vel_mps, angle_deg in targets:
        tau = 2.0 * range_m / 3e8
        fd = 2.0 * vel_mps / config.wavelength_m
        angle_rad = np.deg2rad(angle_deg)
        for rx_idx in range(nr):
            phase_steering = (
                2 * np.pi * rx_pos[rx_idx] * np.sin(angle_rad) / config.wavelength_m
            )
            beat_phase = (
                2 * np.pi * config.chirp_slope * t_fast[:, None] * tau
                + 2 * np.pi * fd * (t_slow[None, :] + tau)
                + phase_steering
            )
            cube[:, :, rx_idx] += np.cos(beat_phase)

    signal_power = np.mean(cube**2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = rng.normal(0, np.sqrt(noise_power), cube.shape).astype(np.float32)
    return cube + noise


def apply_rf_front_end(
    cube: np.ndarray,
    *,
    sample_rate_hz: float,
    hpf_hz: float = 0.0,
    lpf_hz: float | None = None,
    tx_power_db: float = _TX_PWR_REF_DB,
    rx_gain_db: float = _RX_GAIN_REF_DB,
) -> np.ndarray:
    """
    Apply TX/RX gain and HPF/LPF to a real ADC cube (fast-time axis).

    Amplitude is scaled relative to TX=0 dB and RX=26 dB. Band-limiting uses
    an FFT mask on the sample (beat-frequency) axis.
    """
    scale = 10 ** ((tx_power_db - _TX_PWR_REF_DB) / 20.0) * 10 ** (
        (rx_gain_db - _RX_GAIN_REF_DB) / 20.0
    )
    out = cube.astype(np.float64, copy=True) * float(scale)

    ns = out.shape[0]
    fs = float(sample_rate_hz)
    nyquist = 0.5 * fs
    hpf = max(0.0, float(hpf_hz))
    lpf = float(lpf_hz) if lpf_hz is not None else nyquist
    lpf = max(hpf, min(lpf, nyquist))

    # Skip FFT work when the passband is the full Nyquist range.
    if hpf <= 0.0 and lpf >= nyquist - 1.0:
        return (out * 1.0).astype(np.float32)

    spec = np.fft.rfft(out, axis=0)
    freqs = np.fft.rfftfreq(ns, d=1.0 / fs)
    mask = (freqs >= hpf) & (freqs <= lpf)
    spec *= mask[:, None, None]
    filtered = np.fft.irfft(spec, n=ns, axis=0)
    return filtered.astype(np.float32)


@RadarDeviceFactory.register("synthetic")
class SyntheticDevice(RadarDevice):
    """
    Factory-registered synthetic radar device.

    Owns ``generate_synthetic_cube`` and streams moving-target scenes for the
    live GUI. RF controls (HPF / LPF / TX power / RX gain) are applied to each
    generated cube so the Range FFT plot reacts immediately.
    """

    device_type = "synthetic"
    description = "Synthetic FMCW cube generator (no hardware)"

    def __init__(
        self,
        config: RadarConfig,
        *,
        config_path: str | Path | None = None,
        snr_db: float = 25.0,
        seed: int = 42,
        targets: list[tuple[float, float, float]] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, config_path=config_path, **kwargs)
        self.snr_db = float(snr_db)
        self.base_targets = targets  # None → moving default scene
        self._rng = np.random.default_rng(seed)
        self._temp_c = 42.0  # simulated die temperature (°C)
        # Optional HW-like programming delays so the GUI busy/revert path is
        # exercised without real silicon (TX power ≥ 1 s by default).
        if kwargs.get("simulate_rf_latency", True):
            self.rf_program_latency_s = {
                "pwr": float(kwargs.get("pwr_latency_s", 1.0)),
                "gain": float(kwargs.get("gain_latency_s", 0.2)),
                "hpf": float(kwargs.get("hpf_latency_s", 0.1)),
                "lpf": float(kwargs.get("lpf_latency_s", 0.1)),
                "chirp_bw": float(kwargs.get("chirp_bw_latency_s", 0.15)),
                "samples": float(kwargs.get("samples_latency_s", 0.15)),
                "sample_rate": float(kwargs.get("sample_rate_latency_s", 0.15)),
                "vmax": float(kwargs.get("vmax_latency_s", 0.15)),
                "dv": float(kwargs.get("dv_latency_s", 0.1)),
            }

    @classmethod
    def default_config_path(cls) -> Path | None:
        """Sample XML shipped with the project, when available."""
        return project_sample_config_path()

    def open(self) -> None:
        """No hardware — mark the synthetic stream as ready."""
        self._opened = True

    def close(self) -> None:
        """No hardware — mark the synthetic stream as stopped."""
        self._opened = False

    def supports_temperature(self) -> bool:
        return True

    def read_temperature(self) -> float | None:
        """
        Simulated real-time temperature (°C).

        Slow thermal drift plus small noise — stand-in for a device sensor.
        """
        if not self._opened:
            return None
        self._temp_c += float(self._rng.normal(0.0, 0.05))
        self._temp_c = float(np.clip(self._temp_c, 35.0, 85.0))
        return self._temp_c

    def read_frame(self, frame_id: int = 0) -> np.ndarray:
        """
        Return a synthetic cube with current RF controls applied.

        HPF / LPF band-limit the beat spectrum; TX power and RX gain scale
        amplitude relative to 0 dB / 26 dB.
        """
        if not self._opened:
            raise RuntimeError("SyntheticDevice is not open — call open() first")

        if self.base_targets is not None:
            targets = self.base_targets
        else:
            phase = frame_id * 0.08
            targets = [
                (15.0 + 2.0 * np.sin(phase), 3.0 + 0.5 * np.cos(phase * 0.7), 10.0),
                (40.0 + 1.5 * np.cos(phase * 0.5), -5.0 + 0.4 * np.sin(phase), -20.0),
            ]
        seed = int(self._rng.integers(0, 2**31 - 1))
        cube = generate_synthetic_cube(
            self.config, targets=targets, snr_db=self.snr_db, seed=seed
        )
        return apply_rf_front_end(
            cube,
            sample_rate_hz=self.config.adc_sample_rate_hz,
            hpf_hz=self.hpf_cutoff_hz,
            lpf_hz=self.lpf_cutoff_hz,
            tx_power_db=self.tx_power_db,
            rx_gain_db=self.rx_gain_db,
        )
