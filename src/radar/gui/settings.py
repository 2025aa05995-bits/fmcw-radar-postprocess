"""Non-blocking device-settings apply with busy discard / revert."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    import tkinter as tk
    from ..devices import RadarDevice


EqualFn = Callable[[Any, Any], bool]
ApplyFn = Callable[["RadarDevice"], None]
UiFn = Callable[[], None]


def _default_equal(a: Any, b: Any) -> bool:
    try:
        return abs(float(a) - float(b)) < 1e-9
    except (TypeError, ValueError):
        return a == b


@dataclass
class TrackedSetting:
    """One GUI setting mirrored to the device."""

    key: str
    label: str
    value: Any
    unit: str = ""


@dataclass
class DeviceSettingsController:
    """
    Track GUI radar settings and apply them off the UI thread.

    While a device ``update*`` call is in flight the controller is *busy*.
    Further user changes are discarded and the UI is restored to the last
    committed value so Tk never blocks on slow hardware programming
    (e.g. TX power ≥ 1 s).
    """

    root: tk.Misc
    device_getter: Callable[[], RadarDevice | None]
    on_status: Callable[[str], None] | None = None

    _committed: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _labels: dict[str, str] = field(default_factory=dict, init=False, repr=False)
    _units: dict[str, str] = field(default_factory=dict, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _busy: bool = field(default=False, init=False)
    _busy_key: str | None = field(default=None, init=False)
    _busy_label: str = field(default="", init=False)
    _status: str = field(default="Idle", init=False)
    _detail: str = field(default="", init=False)

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._busy

    @property
    def status_text(self) -> str:
        with self._lock:
            return self._status

    @property
    def detail_text(self) -> str:
        with self._lock:
            return self._detail

    @property
    def busy_label(self) -> str:
        with self._lock:
            return self._busy_label

    def committed(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._committed.get(key, default)

    def snapshot(self) -> list[TrackedSetting]:
        """Return committed settings for the status panel (display only)."""
        with self._lock:
            return [
                TrackedSetting(
                    key=k,
                    label=self._labels.get(k, k),
                    value=v,
                    unit=self._units.get(k, ""),
                )
                for k, v in self._committed.items()
            ]

    def format_snapshot(self) -> str:
        """Compact one-line summary of tracked settings."""
        parts: list[str] = []
        for s in self.snapshot():
            val = s.value
            if isinstance(val, float):
                text = f"{val:g}"
            else:
                text = str(val)
            unit = f" {s.unit}" if s.unit else ""
            parts.append(f"{s.label}={text}{unit}")
        return "  ·  ".join(parts) if parts else "(none)"

    def seed(
        self,
        key: str,
        value: Any,
        *,
        label: str | None = None,
        unit: str = "",
    ) -> None:
        """Record a setting as committed without calling the device."""
        with self._lock:
            self._committed[key] = value
            if label:
                self._labels[key] = label
            if unit:
                self._units[key] = unit

    def clear(self) -> None:
        """Drop tracked values (e.g. after a device swap)."""
        with self._lock:
            self._committed.clear()
            self._labels.clear()
            self._units.clear()
            self._busy = False
            self._busy_key = None
            self._busy_label = ""
            self._status = "Idle"
            self._detail = ""

    def request(
        self,
        key: str,
        value: Any,
        *,
        label: str,
        apply: ApplyFn,
        unit: str = "",
        equal: EqualFn | None = None,
        on_applied: Callable[[Any], None] | None = None,
        on_discarded: Callable[[Any], None] | None = None,
        on_error: Callable[[BaseException], None] | None = None,
    ) -> bool:
        """
        Try to apply ``value`` for ``key`` on a background thread.

        Returns
        -------
        bool
            ``True`` if the apply was started, ``False`` if discarded
            (device busy, or value already committed).
        """
        eq = equal or _default_equal
        with self._lock:
            if label:
                self._labels[key] = label
            if unit:
                self._units[key] = unit

            if self._busy:
                committed = self._committed.get(key)
                busy_lbl = self._busy_label or "device"
                self._status = f"Busy ({busy_lbl}) — discarded {label}"
                self._detail = "Restored last applied value"
                status = self._status
                detail = self._detail
                self._emit_status(status, detail)
                if on_discarded is not None and key in self._committed:
                    self._ui(lambda c=committed, cb=on_discarded: cb(c))
                return False

            if key in self._committed and eq(self._committed[key], value):
                return False

            previous = self._committed.get(key)
            self._busy = True
            self._busy_key = key
            self._busy_label = label
            self._status = f"Applying {label}…"
            if previous is None:
                self._detail = f"→ {self._fmt(value, unit)}"
            else:
                self._detail = (
                    f"{self._fmt(previous, unit)} → {self._fmt(value, unit)}"
                )
            status = self._status
            detail = self._detail

        self._emit_status(status, detail)

        def work() -> None:
            err: BaseException | None = None
            try:
                device = self.device_getter()
                if device is None:
                    raise RuntimeError("no device")
                apply(device)
            except BaseException as exc:  # noqa: BLE001
                err = exc

            with self._lock:
                self._busy = False
                self._busy_key = None
                self._busy_label = ""
                if err is None:
                    self._committed[key] = value
                    self._status = f"Applied {label}"
                    self._detail = self._fmt(value, unit)
                else:
                    self._status = f"Failed {label}"
                    self._detail = str(err)
                status_now = self._status
                detail_now = self._detail
                committed_now = self._committed.get(key)

            self._emit_status(status_now, detail_now)
            if err is None:
                if on_applied is not None:
                    self._ui(lambda v=value, cb=on_applied: cb(v))
            else:
                # Revert UI to last good committed value when apply fails.
                if on_discarded is not None and committed_now is not None:
                    self._ui(lambda c=committed_now, cb=on_discarded: cb(c))
                if on_error is not None:
                    self._ui(lambda e=err, cb=on_error: cb(e))

        threading.Thread(
            target=work,
            name=f"DeviceSettings:{key}",
            daemon=True,
        ).start()
        return True

    @staticmethod
    def _fmt(value: Any, unit: str) -> str:
        if isinstance(value, float):
            text = f"{value:g}"
        else:
            text = str(value)
        return f"{text} {unit}".strip() if unit else text

    def _emit_status(self, status: str, detail: str) -> None:
        if self.on_status is None:
            return

        def notify() -> None:
            if self.on_status is not None:
                msg = status if not detail else f"{status}  ·  {detail}"
                self.on_status(msg)

        self._ui(notify)

    def _ui(self, fn: UiFn) -> None:
        """Marshal a callback onto the Tk main thread."""
        try:
            self.root.after(0, fn)
        except Exception:  # noqa: BLE001
            pass
