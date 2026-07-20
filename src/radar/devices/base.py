"""Radar configuration, cube I/O, and abstract device interface."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
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
    exposed as read-only properties. Shared by every ``RadarDevice``.
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
        from .synthetic import generate_synthetic_cube

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
        from .synthetic import generate_synthetic_cube

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


def project_sample_config_path() -> Path | None:
    """Locate ``data/sample/radar_config.xml`` relative to the repo when present."""
    # devices/ -> radar/ -> src/ -> project root
    root = Path(__file__).resolve().parents[3]
    sample = root / "data" / "sample" / "radar_config.xml"
    return sample if sample.is_file() else None


class RadarDevice(ABC):
    """
    Abstract base for a radar capture device / driver.

    Every device holds a ``RadarConfig`` (chirp / ADC / RF parameters). Subclass
    ``DummyDevice`` (see ``dummy.py``) when adding real hardware — register with
    ``RadarDeviceFactory`` and override ``open`` / ``close`` / ``read_frame``.

    Example::

        @RadarDeviceFactory.register("my_radar")
        class MyRadar(RadarDevice):
            device_type = "my_radar"
            description = "Acme USB radar"

            def open(self) -> None: ...
            def close(self) -> None: ...
            def read_frame(self, frame_id: int = 0) -> np.ndarray: ...
    """

    #: Short identifier used by the factory (override in subclasses).
    device_type: str = "device"
    #: One-line description shown in device pickers / CLI help.
    description: str = "Radar device"

    def __init__(
        self,
        config: RadarConfig,
        *,
        config_path: str | Path | None = None,
        **kwargs: Any,
    ) -> None:
        self.config = config
        self.config_path = Path(config_path) if config_path is not None else None
        self._opened = False
        self._device_options: dict[str, Any] = dict(kwargs)
        #: High-pass filter cutoff in Hz (set via ``updateHpf``).
        self.hpf_cutoff_hz: float = 0.0
        #: Low-pass filter cutoff in Hz (set via ``updateLpf``).
        self.lpf_cutoff_hz: float = 15_000_000.0
        #: TX power setting in dB (set via ``updatePwr``).
        self.tx_power_db: float = 0.0
        #: RX gain setting in dB (set via ``updateGain``).
        self.rx_gain_db: float = 26.0
        #: Chirp sweep bandwidth in Hz (set via ``updateChirpBw``).
        self.chirp_bandwidth_hz: float = float(config.bandwidth_hz)
        #: ADC samples per chirp (set via ``updateSamples``).
        self.num_samples: int = int(config.num_samples)
        #: ADC sample rate in Hz (set via ``updateSampleRate``).
        self.adc_sample_rate_hz: float = float(config.adc_sample_rate_hz)
        #: Chirps per frame (set via ``updateVelocityResolution``).
        self.num_chirps: int = int(config.num_chirps)
        #: Optional callback when RF controls change (e.g. live-worker nudge).
        self._controls_changed_cb: Any = None

    @classmethod
    def default_config_path(cls) -> Path | None:
        """
        Optional default XML / config file for this device type.

        Override for hardware that ships with a known profile path.
        """
        return None

    @classmethod
    def load_device_config(
        cls,
        *,
        config: RadarConfig | None = None,
        config_path: str | Path | None = None,
        **kwargs: Any,
    ) -> RadarConfig:
        """
        Resolve ``RadarConfig`` for this device.

        Priority: explicit ``config`` → ``config_path`` → ``default_config_path()``
        → built-in ``RadarConfig()``. Real drivers can override to merge SDK /
        EEPROM parameters into the config object.
        """
        if config is not None:
            return config
        path = config_path if config_path is not None else cls.default_config_path()
        if path is not None:
            return load_config(path)
        return RadarConfig()

    @classmethod
    def create(
        cls,
        *,
        config: RadarConfig | None = None,
        config_path: str | Path | None = None,
        **kwargs: Any,
    ) -> RadarDevice:
        """
        Build a configured device instance (config + driver options).

        Prefer ``RadarDeviceFactory.create(...)`` from application code so the
        device type is selected by name.
        """
        resolved_path = config_path
        if resolved_path is None and config is None:
            resolved_path = cls.default_config_path()
        cfg = cls.load_device_config(
            config=config, config_path=resolved_path, **kwargs
        )
        return cls(cfg, config_path=resolved_path, **kwargs)

    @property
    def name(self) -> str:
        """Human-readable name shown in the GUI status bar."""
        return self.device_type

    @property
    def is_open(self) -> bool:
        """True after a successful ``open()`` until ``close()``."""
        return self._opened

    @abstractmethod
    def open(self) -> None:
        """
        Open / connect the device and prepare for streaming.

        Must set ``self._opened = True`` on success.
        """

    @abstractmethod
    def close(self) -> None:
        """
        Stop streaming and release device resources.

        Must set ``self._opened = False``. Safe to call if already closed.
        """

    @abstractmethod
    def read_frame(self, frame_id: int = 0) -> np.ndarray:
        """
        Capture one radar cube.

        Returns
        -------
        np.ndarray
            Real ADC cube with shape ``(num_samples, num_chirps, num_rx)``.
        """

    def supports_temperature(self) -> bool:
        """Return True if this device can report real-time temperature."""
        return False

    def read_temperature(self) -> float | None:
        """
        Read device temperature in degrees Celsius.

        Override in drivers that expose a thermal sensor. Default returns
        ``None`` (no temperature available).
        """
        return None

    def updateHpf(self, cutoff_hz: float) -> None:
        """
        Update the device high-pass filter cutoff frequency.

        Parameters
        ----------
        cutoff_hz :
            Cutoff in Hz (GUI typically steps 200–3600 kHz). Real drivers
            should program the hardware; the default stores the value on
            ``self.hpf_cutoff_hz``.
        """
        self.hpf_cutoff_hz = float(cutoff_hz)
        self._notify_controls_changed()

    def updateLpf(self, cutoff_hz: float) -> None:
        """
        Update the device low-pass filter cutoff frequency.

        Parameters
        ----------
        cutoff_hz :
            Cutoff in Hz (GUI typically steps 15–40 MHz). Real drivers
            should program the hardware; the default stores the value on
            ``self.lpf_cutoff_hz``.
        """
        self.lpf_cutoff_hz = float(cutoff_hz)
        self._notify_controls_changed()

    def updatePwr(self, power_db: float) -> None:
        """
        Update TX transmit power.

        Parameters
        ----------
        power_db :
            TX power in dB (GUI: 0–15, 1 dB steps). Real drivers should
            program the PA; default stores ``self.tx_power_db``.
        """
        self.tx_power_db = float(power_db)
        self._notify_controls_changed()

    def updateGain(self, gain_db: float) -> None:
        """
        Update RX gain.

        Parameters
        ----------
        gain_db :
            RX gain in dB (GUI: 26–48, 3 dB steps). Real drivers should
            program the LNA/VGA; default stores ``self.rx_gain_db``.
        """
        self.rx_gain_db = float(gain_db)
        self._notify_controls_changed()

    def updateChirpBw(self, bandwidth_hz: float) -> None:
        """
        Update chirp sweep bandwidth (sets range resolution).

        Parameters
        ----------
        bandwidth_hz :
            Chirp RF bandwidth in Hz. Updates ``self.chirp_bandwidth_hz`` and
            ``self.config.bandwidth_hz`` (and clears an explicit chirp slope so
            slope tracks bandwidth / ramp time). Real drivers should program
            the chirp synthesizer.
        """
        bw = float(bandwidth_hz)
        if bw <= 0.0:
            raise ValueError("chirp bandwidth must be positive")
        self.chirp_bandwidth_hz = bw
        self.config.bandwidth_hz = bw
        # Keep slope consistent with BW / ramp unless vendor sets it explicitly later.
        self.config.chirp_slope_hz_per_s = None
        self._notify_controls_changed()

    def updateSamples(self, num_samples: int) -> None:
        """
        Update the number of ADC samples per chirp.

        Parameters
        ----------
        num_samples :
            Fast-time sample count (GUI: 128 / 512 / 1024 / 2048 / 4096).
            Updates ``self.num_samples`` and ``self.config.num_samples``.
            Real drivers should reprogram the ADC / chirp profile.
        """
        n = int(num_samples)
        if n <= 0:
            raise ValueError("num_samples must be positive")
        self.num_samples = n
        self.config.num_samples = n
        self._notify_controls_changed()

    def updateSampleRate(self, sample_rate_hz: float) -> None:
        """
        Update the ADC sampling rate.

        Parameters
        ----------
        sample_rate_hz :
            Sample rate in Hz (GUI: 20e6 / 40e6 / 80e6 for 20 / 40 / 80 MSPS).
            Updates ``self.adc_sample_rate_hz`` and ``self.config.adc_sample_rate_hz``.
            Real drivers should reprogram the ADC clock / profile.
        """
        fs = float(sample_rate_hz)
        if fs <= 0.0:
            raise ValueError("sample rate must be positive")
        self.adc_sample_rate_hz = fs
        self.config.adc_sample_rate_hz = fs
        self._notify_controls_changed()

    def updateMaxVelocity(self, v_max_mps: float) -> None:
        """
        Update maximum unambiguous velocity by programming the chirp period.

        Maps ``v_max = λ / (4 · T_c)`` → ``T_c = λ / (4 · v_max)``. Prefers
        adjusting ``idle_time_s``; shortens ``chirp_duration_s`` only when
        ``T_c`` would otherwise fall below the ramp time.

        Parameters
        ----------
        v_max_mps :
            Desired max unambiguous velocity in m/s (one-sided).
        """
        v = float(v_max_mps)
        if v <= 0.0:
            raise ValueError("max velocity must be positive")
        t_c = self.config.wavelength_m / (4.0 * v)
        ramp = float(self.config.chirp_duration_s)
        if t_c < ramp:
            self.config.chirp_duration_s = t_c
            self.config.idle_time_s = 0.0
            # Slope was derived from BW / ramp — clear explicit override.
            self.config.chirp_slope_hz_per_s = None
        else:
            self.config.idle_time_s = t_c - ramp
        self._notify_controls_changed()

    def updateVelocityResolution(self, dv_mps: float) -> None:
        """
        Update velocity resolution by programming chirps-per-frame.

        Maps ``Δv = λ / (2 · N · T_c)`` → ``N = round(λ / (2 · Δv · T_c))``,
        clamped to ``64…1024`` chirps.

        Parameters
        ----------
        dv_mps :
            Desired velocity resolution in m/s.
        """
        dv = float(dv_mps)
        if dv <= 0.0:
            raise ValueError("velocity resolution must be positive")
        t_c = float(self.config.chirp_period_s)
        if t_c <= 0.0:
            raise ValueError("chirp period must be positive")
        n = int(round(self.config.wavelength_m / (2.0 * dv * t_c)))
        n = max(64, min(n, 1024))
        self.num_chirps = n
        self.config.num_chirps = n
        self._notify_controls_changed()

    def set_controls_changed_callback(self, callback: Any) -> None:
        """Register a callback invoked after any RF control update."""
        self._controls_changed_cb = callback

    def _notify_controls_changed(self) -> None:
        cb = self._controls_changed_cb
        if callable(cb):
            cb()

    def __enter__(self) -> RadarDevice:
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
