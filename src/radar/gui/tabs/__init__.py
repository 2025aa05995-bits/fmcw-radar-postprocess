"""Built-in measurement tabs — import this package to register them."""

from . import angle, cfar, qa, range_doppler, range_fft, raw_adc, temperature
from .base import MeasurementTab, clear_registry, register_tab, registered_tabs

__all__ = [
    "MeasurementTab",
    "register_tab",
    "registered_tabs",
    "clear_registry",
]
