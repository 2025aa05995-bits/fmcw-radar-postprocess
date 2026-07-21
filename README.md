# FMCW Radar Data Cube — Offline Post-Processing

Python notebook, library, and **live GUI** for FMCW radar post-processing, targeting **NXP SAF85xx** radar captures.

## Data format

| Axis | Description |
|------|-------------|
| **samples** | Fast-time ADC samples per chirp |
| **chirps** | Slow-time chirps per frame |
| **rx** | Receive antenna channels |

Raw cube layout: `(num_samples, num_chirps, num_rx)` — real ADC samples (one value per fast-time sample).

Radar configuration is read from an accompanying **XML file** (see `data/sample/radar_config.xml`).

## Quick start

```powershell
cd C:\Users\vikra\Projects\fmcw-radar-postprocess
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
jupyter notebook notebooks/fmcw_postprocess.ipynb
```

## Live GUI

Continuous live viewer with a tab per measurement. Frames come from a
**device driver** created by ``RadarDeviceFactory`` (default: ``synthetic``
— no hardware required).

```powershell
# From project root
$env:PYTHONPATH = "src"
python -m radar.gui
# or: run_gui.bat

python -m radar.gui --config data/sample/radar_config.xml --device synthetic
python -m radar.gui --device synthetic --interval-ms 100
```

Built-in tabs: **Raw ADC**, **Range FFT** (chirp / avg / max-hold), **Range-Doppler**, **CFAR**, **Angle**, **QA** (power, correlation, group delay).

Radar parameter changes (TX power, filters, sample rate, …) are applied on a
**background thread**. The bottom **Status** section shows activity, tracked
settings, and stream state. While the device is busy (TX power programming is
typically ≥ 1 s), further setting edits are **discarded** and controls snap
back to the last applied value so the GUI stays responsive.

### Adding a device driver

Copy ``src/radar/devices/dummy.py`` (the template) rather than
``synthetic.py``. ``DummyDevice`` already wires ``RadarConfig`` loading and
shows where to plug in SDK / USB calls.

1. Copy `dummy.py` → `my_radar.py` and rename the class / factory key
2. Fill in ``TODO`` sections: ``open`` / ``close`` / ``read_frame``
   (and optional ``read_temperature`` / EEPROM merge in ``load_device_config``)
3. Import the module from `devices/__init__.py` so it registers

```python
from pathlib import Path
import numpy as np
from .base import RadarConfig, RadarDevice, load_config
from .factory import RadarDeviceFactory

@RadarDeviceFactory.register("my_radar")
class MyRadarDevice(RadarDevice):
    device_type = "my_radar"
    description = "Acme USB radar"

    @classmethod
    def default_config_path(cls) -> Path | None:
        return Path("data/sample/my_radar.xml")

    @classmethod
    def load_device_config(cls, *, config=None, config_path=None, **kwargs):
        # optional: merge EEPROM / SDK defaults into RadarConfig
        return super().load_device_config(
            config=config, config_path=config_path, **kwargs
        )

    def open(self) -> None:
        # TODO: VendorSDK.open(...)
        self._opened = True

    def close(self) -> None:
        self._opened = False

    def read_frame(self, frame_id: int = 0) -> np.ndarray:
        # TODO: fetch ADC → shape (num_samples, num_chirps, num_rx)
        ...
```

Built-in devices: ``synthetic`` (moving targets for demos), ``dummy`` (zero
cubes + driver template).

Create via the factory only (GUI / CLI already do this):

```python
from radar.devices import RadarDeviceFactory
device = RadarDeviceFactory.create("my_radar", config_path="...")
```

### Adding a new measurement tab

1. Create `src/radar/gui/tabs/my_metric.py`
2. Subclass `MeasurementTab`, decorate with `@register_tab`
3. Import the module in `src/radar/gui/tabs/__init__.py`

```python
from tkinter import ttk
from ..frame import RadarFrame
from .base import MeasurementTab, register_tab
from .plotting import EmbeddedFigure

@register_tab
class MyMetricTab(MeasurementTab):
    title = "My Metric"
    order = 60  # tab sort order

    def build(self, parent: ttk.Frame) -> None:
        self.plot = EmbeddedFigure(parent)

    def update(self, frame: RadarFrame) -> None:
        if not self.is_visible:
            return
        # use frame.cube / frame.range_cube / frame.rd_map / frame.qa_report
        self.plot.draw_idle()
```

No changes to `app.py` are required — tabs are auto-discovered.

## Project layout

```
fmcw-radar-postprocess/
├── notebooks/fmcw_postprocess.ipynb   # Offline pipeline notebook
├── run_gui.bat                        # Launch live GUI
├── src/radar/                         # Processing library + live GUI
│   ├── process.py                     # FFT + CFAR + pipeline
│   ├── testing.py                     # Raw ADC QA
│   ├── visualization.py               # Matplotlib / Plotly plots
│   ├── devices/                       # Device drivers + config + cube I/O
│   │   ├── base.py                    # RadarConfig, cube I/O, RadarDevice ABC
│   │   ├── synthetic.py               # generate_synthetic_cube + SyntheticDevice
│   │   ├── dummy.py                   # DummyDevice template for real hardware
│   │   └── factory.py                 # RadarDeviceFactory
│   └── gui/                           # Continuous live measurement GUI
│       ├── app.py                     # Main window
│       ├── live.py                    # Background worker
│       ├── frame.py                   # Live frame + cached FFTs
│       └── tabs/                      # Pluggable measurement tabs
└── data/sample/                       # Sample config
```

## Using your own data

1. Place your raw capture (`.bin`, `.raw`, or `.npy`) in `data/raw/`.
2. Place your radar XML config alongside it.
3. Open the notebook and set `RAW_PATH` / `CONFIG_PATH`, or launch the GUI with `--config`.

If your XML uses different tag names, edit the alias map in `src/radar/devices/base.py`.

## Processing pipeline

1. **Load** raw cube + parse XML configuration
2. **Range FFT** — distance axis
3. **Doppler FFT** — velocity axis (Range-Doppler map)
4. **Angle FFT** — azimuth axis (requires RX array geometry)
5. **CFAR** — 2D cell-averaging detection on Range-Doppler
6. **Visualize** — heatmaps, detected target overlay, or live GUI tabs
