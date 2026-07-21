"""Measurement tab plugin base and registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, TypeVar

import tkinter as tk
from tkinter import ttk

from ..frame import RadarFrame

if TYPE_CHECKING:
    from ...devices import RadarDevice
    from ..settings import DeviceSettingsController

TabCls = TypeVar("TabCls", bound=type)

_REGISTRY: list[type[MeasurementTab]] = []


def register_tab(cls: TabCls) -> TabCls:
    """
    Class decorator — register a measurement tab for auto-discovery.

    Example::

        @register_tab
        class MyTab(MeasurementTab):
            title = "My Measurement"
            order = 40
            ...
    """
    if not issubclass(cls, MeasurementTab):
        raise TypeError(f"{cls!r} must subclass MeasurementTab")
    _REGISTRY.append(cls)
    return cls


def registered_tabs() -> list[type[MeasurementTab]]:
    """Return registered tab classes sorted by ``order`` then title."""
    return sorted(_REGISTRY, key=lambda c: (getattr(c, "order", 100), c.title))


def clear_registry() -> None:
    """Clear registered tabs (tests / reload)."""
    _REGISTRY.clear()


class MeasurementTab(ABC):
    """
    Base class for one measurement notebook tab.

    Subclass, set ``title`` / ``order``, implement ``build`` + ``update``,
    and decorate with ``@register_tab``. The main app discovers tabs
    automatically — no edits to ``app.py`` required.

    Set ``needs_background_ingest = True`` only when the tab must collect
    data while hidden (e.g. temperature history). Display work stays in
    ``update`` and runs only for the active tab.
    """

    title: str = "Untitled"
    order: int = 100
    #: When False (default), the app skips ``ingest_frame`` while hidden.
    needs_background_ingest: bool = False

    def __init__(self) -> None:
        self.root: tk.Misc | None = None
        self.frame: ttk.Frame | None = None
        self._visible = False
        self._device_getter: Callable[[], RadarDevice | None] | None = None
        self._settings: DeviceSettingsController | None = None
        self._suppress_setting_events = False

    def bind_device(self, getter: Callable[[], RadarDevice | None]) -> None:
        """
        Attach a live device accessor from the main app.

        ``getter`` should return the current ``RadarDevice`` (e.g. the worker's
        device) so tabs can call driver methods such as ``updateHpf``.
        """
        self._device_getter = getter

    def bind_settings(self, settings: DeviceSettingsController) -> None:
        """Attach the shared non-blocking settings controller."""
        self._settings = settings

    @property
    def settings(self) -> DeviceSettingsController | None:
        """Shared device-settings controller, or ``None`` if not bound."""
        return self._settings

    @property
    def device(self) -> RadarDevice | None:
        """Current live device, or ``None`` if not bound / unavailable."""
        if self._device_getter is None:
            return None
        return self._device_getter()

    def attach(self, notebook: ttk.Notebook) -> ttk.Frame:
        """Create the tab page and register it on the notebook."""
        self.root = notebook
        page = ttk.Frame(notebook, padding=6)
        self.frame = page
        self.build(page)
        notebook.add(page, text=self.title)
        return page

    @abstractmethod
    def build(self, parent: ttk.Frame) -> None:
        """Create widgets inside ``parent`` (called once)."""

    @abstractmethod
    def update(self, frame: RadarFrame) -> None:
        """Refresh the tab with a new live frame (called on the UI thread)."""

    def ingest_frame(self, frame: RadarFrame) -> None:
        """
        Optional background ingest while another tab is active.

        Override to keep collecting data (e.g. temperature history) without
        redrawing. Only called when ``needs_background_ingest`` is True
        (or the tab is visible).
        """
        return

    def on_show(self) -> None:
        """Called when this tab becomes the selected notebook page."""
        self._visible = True

    def on_hide(self) -> None:
        """Called when another tab is selected."""
        self._visible = False

    def on_device_changed(self) -> None:
        """
        Called after the live device is swapped in the main app.

        Override to re-sync tab controls from the new driver (seed settings;
        do not block the UI with device ``update*`` calls).
        """
        return

    @property
    def is_visible(self) -> bool:
        return self._visible

    def teardown(self) -> None:
        """Optional cleanup when the app closes."""
        return
