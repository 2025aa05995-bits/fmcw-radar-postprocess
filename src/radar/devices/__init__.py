"""Radar device drivers — config, I/O, abstract base, synthetic / dummy, factory."""

from .base import (
    SPEED_OF_LIGHT,
    RadarConfig,
    RadarDataCube,
    RadarDataIO,
    RadarDevice,
    load_config,
    load_radar_cube,
    project_sample_config_path,
)
from .factory import DeviceInfo, RadarDeviceFactory
from .synthetic import SyntheticDevice, generate_synthetic_cube, apply_rf_front_end
from .dummy import DummyDevice

# Importing SyntheticDevice / DummyDevice registers factory names.

__all__ = [
    "SPEED_OF_LIGHT",
    "RadarConfig",
    "RadarDataCube",
    "RadarDataIO",
    "load_config",
    "load_radar_cube",
    "generate_synthetic_cube",
    "apply_rf_front_end",
    "project_sample_config_path",
    "RadarDevice",
    "SyntheticDevice",
    "DummyDevice",
    "RadarDeviceFactory",
    "DeviceInfo",
]
