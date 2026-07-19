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
    """

    title: str = "Untitled"
    order: int = 100

    def __init__(self) -> None:
        self.root: tk.Misc | None = None
        self.frame: ttk.Frame | None = None
        self._visible = False
        self._device_getter: Callable[[], RadarDevice | None] | None = None

    def bind_device(self, getter: Callable[[], RadarDevice | None]) -> None:
        """
        Attach a live device accessor from the main app.

        ``getter`` should return the current ``RadarDevice`` (e.g. the worker's
        device) so tabs can call driver methods such as ``updateHpf``.
        """
        self._device_getter = getter

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
        redrawing. Called every live frame for all tabs.
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

        Override to re-push tab settings (e.g. HPF) to the new driver.
        """
        return

    @property
    def is_visible(self) -> bool:
        return self._visible

    def teardown(self) -> None:
        """Optional cleanup when the app closes."""
        return
