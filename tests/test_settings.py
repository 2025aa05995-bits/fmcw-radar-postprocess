"""Unit tests for non-blocking device settings apply / busy discard."""

from __future__ import annotations

import threading
import time
import unittest

from radar.gui.settings import DeviceSettingsController


class _FakeRoot:
    """Minimal stand-in for ``tk.Misc.after`` (runs callbacks immediately)."""

    def after(self, _ms: int, fn) -> None:
        fn()


class _FakeDevice:
    def __init__(self) -> None:
        self.tx_power_db = 0.0
        self.calls: list[float] = []

    def updatePwr(self, power_db: float) -> None:
        time.sleep(0.15)
        self.tx_power_db = float(power_db)
        self.calls.append(float(power_db))


class DeviceSettingsControllerTests(unittest.TestCase):
    def test_apply_commits_value(self) -> None:
        device = _FakeDevice()
        statuses: list[str] = []
        ctrl = DeviceSettingsController(
            _FakeRoot(),  # type: ignore[arg-type]
            device_getter=lambda: device,
            on_status=statuses.append,
        )
        ctrl.seed("tx_pwr_db", 0, label="TX power", unit="dB")
        done = threading.Event()
        applied: list[object] = []

        started = ctrl.request(
            "tx_pwr_db",
            5,
            label="TX power",
            unit="dB",
            apply=lambda d: d.updatePwr(5),
            on_applied=lambda v: (applied.append(v), done.set()),
        )
        self.assertTrue(started)
        self.assertTrue(ctrl.busy)
        self.assertTrue(done.wait(timeout=2.0))
        self.assertFalse(ctrl.busy)
        self.assertEqual(ctrl.committed("tx_pwr_db"), 5)
        self.assertEqual(device.tx_power_db, 5.0)
        self.assertEqual(applied, [5])
        self.assertTrue(any("Applying" in s for s in statuses))
        self.assertTrue(any("Applied" in s for s in statuses))

    def test_busy_discards_and_reverts(self) -> None:
        device = _FakeDevice()
        ctrl = DeviceSettingsController(
            _FakeRoot(),  # type: ignore[arg-type]
            device_getter=lambda: device,
        )
        ctrl.seed("tx_pwr_db", 0, label="TX power", unit="dB")
        done = threading.Event()
        discarded: list[object] = []

        self.assertTrue(
            ctrl.request(
                "tx_pwr_db",
                3,
                label="TX power",
                apply=lambda d: d.updatePwr(3),
                on_applied=lambda _v: done.set(),
            )
        )
        # Second change while busy must be discarded + restore committed (0).
        second = ctrl.request(
            "tx_pwr_db",
            9,
            label="TX power",
            apply=lambda d: d.updatePwr(9),
            on_discarded=lambda v: discarded.append(v),
        )
        self.assertFalse(second)
        self.assertEqual(discarded, [0])
        self.assertTrue(done.wait(timeout=2.0))
        self.assertEqual(ctrl.committed("tx_pwr_db"), 3)
        self.assertEqual(device.calls, [3.0])
        self.assertIn("Applied", ctrl.status_text)

    def test_duplicate_value_skipped(self) -> None:
        device = _FakeDevice()
        ctrl = DeviceSettingsController(
            _FakeRoot(),  # type: ignore[arg-type]
            device_getter=lambda: device,
        )
        ctrl.seed("tx_pwr_db", 4, label="TX power", unit="dB")
        started = ctrl.request(
            "tx_pwr_db",
            4,
            label="TX power",
            apply=lambda d: d.updatePwr(4),
        )
        self.assertFalse(started)
        self.assertEqual(device.calls, [])


if __name__ == "__main__":
    unittest.main()
