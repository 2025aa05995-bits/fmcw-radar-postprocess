"""Factory for creating registered radar device drivers with config."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .base import RadarConfig, RadarDevice


@dataclass(frozen=True)
class DeviceInfo:
    """Metadata for a registered device type."""

    name: str
    device_cls: type[RadarDevice]
    description: str


class RadarDeviceFactory:
    """
    Registry / factory for ``RadarDevice`` implementations.

    All live-data setup goes through this factory so applications do not
    hard-code config loading or driver construction::

        device = RadarDeviceFactory.create(
            "synthetic",
            config_path="data/sample/radar_config.xml",
        )
        device.open()
        cube = device.read_frame(0)
    """

    _registry: dict[str, type[RadarDevice]] = {}

    @classmethod
    def register(cls, name: str, device_cls: type[RadarDevice] | None = None):
        """
        Register a device class under ``name``.

        Decorator form::

            @RadarDeviceFactory.register("synthetic")
            class SyntheticDevice(RadarDevice):
                ...
        """
        key = name.strip().lower()

        def _register(device_cls_: type[RadarDevice]) -> type[RadarDevice]:
            if not isinstance(device_cls_, type) or not issubclass(device_cls_, RadarDevice):
                raise TypeError(f"{device_cls_!r} must be a RadarDevice subclass")
            cls._registry[key] = device_cls_
            return device_cls_

        if device_cls is not None:
            return _register(device_cls)
        return _register

    @classmethod
    def unregister(cls, name: str) -> None:
        """Remove a registered device type (tests / hot-reload)."""
        cls._registry.pop(name.strip().lower(), None)

    @classmethod
    def available(cls) -> list[str]:
        """Return registered device type names (sorted)."""
        return sorted(cls._registry.keys())

    @classmethod
    def info(cls, name: str | None = None) -> DeviceInfo | list[DeviceInfo]:
        """Return metadata for one device or all registered devices."""
        if name is not None:
            device_cls = cls.get_class(name)
            return DeviceInfo(
                name=name.strip().lower(),
                device_cls=device_cls,
                description=getattr(device_cls, "description", device_cls.__doc__ or ""),
            )
        return [
            DeviceInfo(
                name=n,
                device_cls=c,
                description=getattr(c, "description", c.__doc__ or "") or n,
            )
            for n, c in sorted(cls._registry.items())
        ]

    @classmethod
    def get_class(cls, name: str) -> type[RadarDevice]:
        """Look up a registered device class by name."""
        key = name.strip().lower()
        if key not in cls._registry:
            known = ", ".join(cls.available()) or "(none)"
            raise KeyError(f"Unknown device type {name!r}. Registered: {known}")
        return cls._registry[key]

    @classmethod
    def load_config(
        cls,
        name: str = "synthetic",
        *,
        config: RadarConfig | None = None,
        config_path: str | Path | None = None,
        **kwargs: Any,
    ) -> RadarConfig:
        """
        Resolve radar configuration via the named device class.

        Delegates to ``DeviceClass.load_device_config`` so each driver can
        apply its own defaults / vendor XML / hardware query.
        """
        device_cls = cls.get_class(name)
        return device_cls.load_device_config(
            config=config, config_path=config_path, **kwargs
        )

    @classmethod
    def create(
        cls,
        name: str = "synthetic",
        *,
        config: RadarConfig | None = None,
        config_path: str | Path | None = None,
        **kwargs: Any,
    ) -> RadarDevice:
        """
        Create a fully configured device by registered name.

        This is the primary application entry point for live data: config
        resolution and driver construction stay inside the device/factory.
        """
        device_cls = cls.get_class(name)
        return device_cls.create(
            config=config, config_path=config_path, **kwargs
        )

    # Backward-compatible alias
    create_from_config = create
