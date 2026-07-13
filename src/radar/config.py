"""FMCW radar configuration and data cube I/O."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

SPEED_OF_LIGHT = 299_792_458.0  # m/s

_PARAM_ALIASES: dict[str, list[str]] = {
    "center_freq_hz": [
        "centerFreq", "center_freq", "fc", "startFrequency", "carrierFrequency",
        "CarrierFrequency", "rfCenterFrequency",
    ],
    "bandwidth_hz": [
        "bandwidth", "sweepBandwidth", "chirpBandwidth", "Bandwidth", "rfBandwidth",
    ],
    "chirp_duration_s": [
        "chirpDuration", "rampTime", "ramp_time", "chirp_time", "ChirpDuration",
        "adcStartTime", "rampEndTime",
    ],
    "adc_sample_rate_hz": [
        "adcSampleRate", "sampleRate", "adc_sample_rate", "AdcSampleRate",
        "fs", "samplingFrequency",
    ],
    "num_samples": [
        "numAdcSamples", "num_samples", "adcSamples", "samples", "NumSamples",
        "numRangeBins", "rangeBins",
    ],
    "num_chirps": [
        "numChirps", "num_chirps", "chirps", "chirpsPerFrame", "NumChirps",
        "numDopplerBins", "dopplerBins",
    ],
    "num_rx": [
        "numRx", "num_rx", "rxChannels", "numRxAntennas", "rxAntennas", "NumRx",
    ],
    "idle_time_s": [
        "idleTime", "idle_time", "chirpIdleTime", "interChirpTime",
    ],
    "chirp_slope_hz_per_s": [
        "chirpSlope", "chirp_slope", "slope", "ChirpSlope",
    ],
    "range_fft_size": ["rangeFftSize", "range_fft_size", "fftSizeRange"],
    "doppler_fft_size": ["dopplerFftSize", "doppler_fft_size", "fftSizeDoppler"],
    "angle_fft_size": ["angleFftSize", "angle_fft_size", "fftSizeAngle"],
    "rx_spacing_m": [
        "rxSpacing", "rx_spacing", "antennaSpacing", "elementSpacing",
    ],
    "data_dtype": ["dataType", "data_dtype", "adcDataType"],
}

_DTYPE_MAP = {
    "int16": np.int16,
    "int32": np.int32,
    "uint16": np.uint16,
    "float32": np.float32,
    "float64": np.float64,
}


@dataclass
class RadarConfig:
    """
    FMCW radar frame configuration.

    Holds chirp timing, ADC dimensions, RF parameters, and optional FFT sizes.
    Derived quantities (wavelength, range resolution, max range, etc.) are
    exposed as read-only properties.
    """

    center_freq_hz: float = 77e9
    bandwidth_hz: float = 1e9
    chirp_duration_s: float = 25.6e-6
    adc_sample_rate_hz: float = 40e6
    num_samples: int = 1024
    num_chirps: int = 128
    num_rx: int = 4
    idle_time_s: float = 10e-6
    chirp_slope_hz_per_s: float | None = None
    range_fft_size: int | None = None
    doppler_fft_size: int | None = None
    angle_fft_size: int | None = None
    rx_spacing_m: float = 0.0019
    data_dtype: str = "int16"

    @property
    def wavelength_m(self) -> float:
        """Carrier wavelength in metres (c / fc)."""
        return SPEED_OF_LIGHT / self.center_freq_hz

    @property
    def chirp_slope(self) -> float:
        """Chirp slope in Hz/s; computed from bandwidth / ramp time if not set explicitly."""
        if self.chirp_slope_hz_per_s is not None:
            return self.chirp_slope_hz_per_s
        return self.bandwidth_hz / self.chirp_duration_s

    @property
    def chirp_period_s(self) -> float:
        """Total chirp period including idle time between chirps."""
        return self.chirp_duration_s + self.idle_time_s

    @property
    def range_resolution_m(self) -> float:
        """Theoretical range resolution: c / (2 × bandwidth)."""
        return SPEED_OF_LIGHT / (2.0 * self.bandwidth_hz)

    @property
    def max_range_m(self) -> float:
        """Maximum unambiguous range from ADC sample rate and chirp slope."""
        return (self.adc_sample_rate_hz * SPEED_OF_LIGHT) / (2.0 * self.chirp_slope)

    @property
    def doppler_resolution_mps(self) -> float:
        """Velocity resolution across the full frame (wavelength / 2T_frame)."""
        frame_time = self.num_chirps * self.chirp_period_s
        return self.wavelength_m / (2.0 * frame_time)

    @property
    def max_velocity_mps(self) -> float:
        """Maximum unambiguous velocity (wavelength / 4T_chirp)."""
        return self.wavelength_m / (4.0 * self.chirp_period_s)

    def summary(self) -> str:
        """Return a one-line human-readable summary of key config parameters."""
        return (
            f"RadarConfig({self.num_samples} samples × {self.num_chirps} chirps × "
            f"{self.num_rx} RX | fc={self.center_freq_hz/1e9:.2f} GHz, "
            f"BW={self.bandwidth_hz/1e6:.0f} MHz, "
            f"dR={self.range_resolution_m:.2f} m, "
            f"dv={self.doppler_resolution_mps:.2f} m/s)"
        )

    @classmethod
    def from_xml(cls, path: str | Path) -> RadarConfig:
        """Load configuration from an XML file (delegates to ``load_config``)."""
        return load_config(path)


def _local_name(tag: str) -> str:
    """Strip XML namespace prefix from an element tag name."""
    return tag.split("}")[-1] if "}" in tag else tag


def _parse_value(raw: str) -> int | float | str:
    """Parse a string leaf value as int, float, or plain string."""
    text = raw.strip()
    if not text:
        return text
    try:
        if "." in text or "e" in text.lower():
            return float(text)
        return int(text)
    except ValueError:
        return text


def _collect_params(root: ET.Element) -> dict[str, Any]:
    """Walk an XML tree and flatten leaf text/attributes into a parameter dict."""
    params: dict[str, Any] = {}

    def walk(elem: ET.Element, prefix: str = "") -> None:
        name = _local_name(elem.tag)
        key = f"{prefix}{name}" if not prefix else f"{prefix}.{name}"
        for attr, val in elem.attrib.items():
            params[attr] = _parse_value(val)
            params[f"{name}.{attr}"] = _parse_value(val)
        if elem.text and elem.text.strip():
            params[name] = _parse_value(elem.text.strip())
            params[key] = _parse_value(elem.text.strip())
        for child in elem:
            walk(child, key)

    walk(root)
    return params


def _resolve(params: dict[str, Any], canonical: str) -> Any | None:
    """Look up a canonical parameter name using vendor-specific XML aliases."""
    for alias in _PARAM_ALIASES.get(canonical, [canonical]):
        if alias in params:
            return params[alias]
    return None


def load_config(path: str | Path) -> RadarConfig:
    """
    Load radar configuration from an XML file.

    Supports flexible tag names via ``_PARAM_ALIASES`` so vendor exports
    (NXP, TI, custom) map onto the canonical ``RadarConfig`` fields.
    """
    tree = ET.parse(Path(path))
    params = _collect_params(tree.getroot())
    cfg = RadarConfig()
    field_map = {
        "center_freq_hz": float,
        "bandwidth_hz": float,
        "chirp_duration_s": float,
        "adc_sample_rate_hz": float,
        "num_samples": int,
        "num_chirps": int,
        "num_rx": int,
        "idle_time_s": float,
        "chirp_slope_hz_per_s": float,
        "range_fft_size": int,
        "doppler_fft_size": int,
        "angle_fft_size": int,
        "rx_spacing_m": float,
    }
    for field, caster in field_map.items():
        val = _resolve(params, field)
        if val is not None:
            setattr(cfg, field, caster(val))
    dtype = _resolve(params, "data_dtype")
    if dtype is not None:
        cfg.data_dtype = str(dtype)
    return cfg


def _as_real_cube(cube: np.ndarray) -> np.ndarray:
    """Cast cube to real float32; reject complex-valued input."""
    if np.iscomplexobj(cube):
        raise ValueError("Expected real ADC samples, got complex data")
    return cube.astype(np.float32, copy=False)


@dataclass
class RadarDataCube:
    """
    Real ADC radar cube bundled with its configuration.

    Data layout: ``(num_samples, num_chirps, num_rx)``.
    """

    data: np.ndarray
    config: RadarConfig

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the underlying data array."""
        return self.data.shape

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        config: RadarConfig,
        *,
        offset_bytes: int = 0,
        scale: float = 1.0,
    ) -> RadarDataCube:
        """
        Load a cube from disk and wrap it with the given config.

        See ``load_radar_cube`` for supported file formats.
        """
        return cls(
            data=load_radar_cube(path, config, offset_bytes=offset_bytes, scale=scale),
            config=config,
        )

    @classmethod
    def synthetic(
        cls,
        config: RadarConfig,
        *,
        targets: list[tuple[float, float, float]] | None = None,
        snr_db: float = 25.0,
        seed: int = 42,
    ) -> RadarDataCube:
        """
        Generate a synthetic cube for testing and return it wrapped with config.

        Each target is ``(range_m, velocity_mps, angle_deg)``.
        """
        return cls(
            data=generate_synthetic_cube(config, targets=targets, snr_db=snr_db, seed=seed),
            config=config,
        )


class RadarDataIO:
    """Static I/O helpers for loading or synthesizing radar cubes as raw arrays."""

    @staticmethod
    def load(
        path: str | Path,
        config: RadarConfig,
        *,
        offset_bytes: int = 0,
        scale: float = 1.0,
    ) -> np.ndarray:
        """
        Load a raw cube array from disk.

        Returns shape ``(samples, chirps, rx)`` as float32.
        """
        return load_radar_cube(path, config, offset_bytes=offset_bytes, scale=scale)

    @staticmethod
    def synthetic(
        config: RadarConfig,
        *,
        targets: list[tuple[float, float, float]] | None = None,
        snr_db: float = 25.0,
        seed: int = 42,
    ) -> np.ndarray:
        """
        Generate a synthetic cube array for pipeline validation.

        Returns shape ``(samples, chirps, rx)`` as float32.
        """
        return generate_synthetic_cube(config, targets=targets, snr_db=snr_db, seed=seed)


def load_radar_cube(
    path: str | Path,
    config: RadarConfig,
    *,
    offset_bytes: int = 0,
    scale: float = 1.0,
) -> np.ndarray:
    """
    Load a raw radar cube and reshape to ``(samples, chirps, rx)``.

    Supports ``.npy`` (pre-shaped 3D) and flat binary ``.bin`` / ``.raw``.
    Binary layout default: ``[rx][chirp][sample]`` interleaved.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".npy":
        cube = np.load(path)
        if cube.ndim != 3:
            raise ValueError(f"Expected 3D .npy cube, got shape {cube.shape}")
        return _as_real_cube(cube) * scale

    dtype = _DTYPE_MAP.get(config.data_dtype, np.int16)
    raw = np.fromfile(path, dtype=dtype, offset=offset_bytes)
    ns, nc, nr = config.num_samples, config.num_chirps, config.num_rx
    expected = ns * nc * nr
    if raw.size < expected:
        raise ValueError(
            f"File has {raw.size} elements, expected at least {expected} "
            f"({ns} samples x {nc} chirps x {nr} rx)"
        )
    raw = raw[:expected]
    cube = np.transpose(raw.reshape(nr, nc, ns), (2, 1, 0))
    return _as_real_cube(cube) * scale


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
            phase_steering = 2 * np.pi * rx_pos[rx_idx] * np.sin(angle_rad) / config.wavelength_m
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
