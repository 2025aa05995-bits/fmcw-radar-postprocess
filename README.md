# FMCW Radar Data Cube — Offline Post-Processing

Python notebook and library for offline FMCW radar post-processing, targeting **NXP SAF85xx** radar captures.

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

## Project layout

```
fmcw-radar-postprocess/
├── notebooks/fmcw_postprocess.ipynb   # Main offline pipeline
├── src/radar/                         # Reusable processing modules
│   ├── config.py                      # XML config parser
│   ├── io.py                          # Raw cube loader
│   ├── processing.py                  # FFT + CFAR pipeline
│   └── visualize.py                   # Plots & heatmaps
└── data/sample/                       # Sample config + synthetic data
```

## Using your own data

1. Place your raw capture (`.bin`, `.raw`, or `.npy`) in `data/raw/`.
2. Place your radar XML config alongside it.
3. Open the notebook and set `RAW_PATH` and `CONFIG_PATH`.

If your XML uses different tag names, edit the alias map in `src/radar/config.py`.

## Processing pipeline

1. **Load** raw cube + parse XML configuration
2. **Range FFT** — distance axis
3. **Doppler FFT** — velocity axis (Range-Doppler map)
4. **Angle FFT** — azimuth axis (requires RX array geometry)
5. **CFAR** — 2D cell-averaging detection on Range-Doppler
6. **Visualize** — heatmaps, detected target overlay
