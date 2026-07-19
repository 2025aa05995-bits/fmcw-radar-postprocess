"""Launch the continuous live measurement GUI.

Usage::

    python -m radar.gui
    python -m radar.gui --config data/sample/radar_config.xml
    python -m radar.gui --device synthetic --interval-ms 100
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


def _project_src() -> Path:
    return Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    # Ensure ``src`` is importable when launched as a script path.
    src = _project_src()
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    from radar.devices import RadarDeviceFactory

    parser = argparse.ArgumentParser(description="FMCW radar live measurement GUI")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Radar XML config path (default: built-in RadarConfig)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="synthetic",
        help=(
            "Device driver name (registered: "
            f"{', '.join(RadarDeviceFactory.available()) or 'synthetic'})"
        ),
    )
    parser.add_argument(
        "--interval-ms",
        type=float,
        default=150.0,
        help="Frame fetch interval in milliseconds (default: 150)",
    )
    args = parser.parse_args(argv)

    config_path = args.config
    if config_path is None:
        sample = Path.cwd() / "data" / "sample" / "radar_config.xml"
        if sample.exists():
            config_path = sample

    from radar.gui.app import launch

    launch(
        config_path=config_path,
        device_type=args.device,
        interval_s=args.interval_ms / 1000.0,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
