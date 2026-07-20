"""
Dummy radar device — template for real hardware drivers.

Copy this module (e.g. to ``my_radar.py``), rename the class, register a new
factory name, and replace the ``TODO`` sections with SDK / USB / SPI calls.
Keep the return shape of ``read_frame`` as ``(num_samples, num_chirps, num_rx)``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .base import RadarConfig, RadarDevice, load_config, project_sample_config_path
from .factory import RadarDeviceFactory


@RadarDeviceFactory.register("dummy")
class DummyDevice(RadarDevice):
    """
    Stub driver used as the pattern for real devices.

    Behaviour today:
    - Loads ``RadarConfig`` from XML (same path resolution as other devices).
    - ``open`` / ``close`` manage a placeholder handle.
    - ``read_frame`` returns a zero-filled cube of the configured shape so the
      GUI / pipeline can smoke-test without hardware.

    Replace the TODO bodies when wiring a real sensor; do not subclass
    ``SyntheticDevice`` for production hardware.
    """

    device_type = "dummy"
    description = "Dummy / template driver (copy to implement real hardware)"

    def __init__(
        self,
        config: RadarConfig,
        *,
        config_path: str | Path | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, config_path=config_path, **kwargs)
        # TODO: store port, baud, SDK options from kwargs
        self._handle: Any = None

    @classmethod
    def default_config_path(cls) -> Path | None:
        """
        Default chirp / ADC profile for this device.

        TODO: point at the vendor XML that ships with your board.
        """
        return project_sample_config_path()

    @classmethod
    def load_device_config(
        cls,
        *,
        config: RadarConfig | None = None,
        config_path: str | Path | None = None,
        **kwargs: Any,
    ) -> RadarConfig:
        """
        Resolve config, optionally merging hardware-reported parameters.

        TODO: after loading XML, query EEPROM / SDK and override fields
        such as ``num_rx``, ``adc_sample_rate_hz``, etc.
        """
        if config is not None:
            return config
        path = config_path if config_path is not None else cls.default_config_path()
        if path is not None:
            cfg = load_config(path)
        else:
            cfg = RadarConfig()
        # Example hook (no-op): cfg.num_rx = kwargs.get("num_rx", cfg.num_rx)
        return cfg

    def open(self) -> None:
        """
        Connect to the device and arm streaming.

        TODO: open USB / SPI / Ethernet / vendor SDK session and store
        the handle on ``self._handle``.
        """
        # self._handle = VendorSDK.open(...)
        self._handle = "dummy-handle"
        self._opened = True

    def close(self) -> None:
        """
        Stop streaming and release resources.

        TODO: call SDK close / release the transport handle.
        """
        # if self._handle is not None:
        #     VendorSDK.close(self._handle)
        self._handle = None
        self._opened = False

    def supports_temperature(self) -> bool:
        """Return True once ``read_temperature`` is implemented for hardware."""
        return False

    def read_temperature(self) -> float | None:
        """
        Read die / board temperature in °C.

        TODO: read sensor register via SDK; return None if unavailable.
        """
        if not self._opened:
            return None
        return None

    def updateHpf(self, cutoff_hz: float) -> None:
        """
        Program the analog/digital HPF cutoff on the device.

        TODO: write the cutoff register / SDK call, e.g.
        ``VendorSDK.set_hpf(self._handle, cutoff_hz)``.
        """
        super().updateHpf(cutoff_hz)
        # VendorSDK.set_hpf(self._handle, cutoff_hz)

    def updateLpf(self, cutoff_hz: float) -> None:
        """
        Program the analog/digital LPF cutoff on the device.

        TODO: ``VendorSDK.set_lpf(self._handle, cutoff_hz)``.
        """
        super().updateLpf(cutoff_hz)
        # VendorSDK.set_lpf(self._handle, cutoff_hz)

    def updatePwr(self, power_db: float) -> None:
        """
        Program TX power on the device.

        TODO: ``VendorSDK.set_tx_power(self._handle, power_db)``.
        """
        super().updatePwr(power_db)
        # VendorSDK.set_tx_power(self._handle, power_db)

    def updateGain(self, gain_db: float) -> None:
        """
        Program RX gain on the device.

        TODO: ``VendorSDK.set_rx_gain(self._handle, gain_db)``.
        """
        super().updateGain(gain_db)
        # VendorSDK.set_rx_gain(self._handle, gain_db)

    def updateChirpBw(self, bandwidth_hz: float) -> None:
        """
        Program chirp bandwidth on the device.

        TODO: ``VendorSDK.set_chirp_bandwidth(self._handle, bandwidth_hz)``.
        """
        super().updateChirpBw(bandwidth_hz)
        # VendorSDK.set_chirp_bandwidth(self._handle, bandwidth_hz)

    def updateSamples(self, num_samples: int) -> None:
        """
        Program ADC samples-per-chirp on the device.

        TODO: ``VendorSDK.set_num_samples(self._handle, num_samples)``.
        """
        super().updateSamples(num_samples)
        # VendorSDK.set_num_samples(self._handle, num_samples)

    def updateSampleRate(self, sample_rate_hz: float) -> None:
        """
        Program ADC sample rate on the device.

        TODO: ``VendorSDK.set_sample_rate(self._handle, sample_rate_hz)``.
        """
        super().updateSampleRate(sample_rate_hz)
        # VendorSDK.set_sample_rate(self._handle, sample_rate_hz)

    def updateMaxVelocity(self, v_max_mps: float) -> None:
        """
        Program chirp period / idle time for the requested max velocity.

        TODO: ``VendorSDK.set_chirp_period(self._handle, t_c)`` or idle time.
        """
        super().updateMaxVelocity(v_max_mps)
        # VendorSDK.set_idle_time(self._handle, self.config.idle_time_s)

    def updateVelocityResolution(self, dv_mps: float) -> None:
        """
        Program chirps-per-frame for the requested velocity resolution.

        TODO: ``VendorSDK.set_num_chirps(self._handle, self.config.num_chirps)``.
        """
        super().updateVelocityResolution(dv_mps)
        # VendorSDK.set_num_chirps(self._handle, self.config.num_chirps)

    def read_frame(self, frame_id: int = 0) -> np.ndarray:
        """
        Capture one ADC cube shaped ``(num_samples, num_chirps, num_rx)``.

        TODO: fetch the raw buffer from the device, cast to float32, and
        reshape / transpose to ``(samples, chirps, rx)``. Until then this
        returns zeros so callers can exercise the pipeline safely.
        """
        if not self._opened:
            raise RuntimeError("DummyDevice is not open — call open() first")

        cfg = self.config
        # Placeholder: real drivers replace this with a hardware capture.
        # Example:
        #   raw = VendorSDK.read_frame(self._handle, frame_id)
        #   return reshape_to_cube(raw, cfg)
        return np.zeros(
            (cfg.num_samples, cfg.num_chirps, cfg.num_rx),
            dtype=np.float32,
        )
