# MAP2 Patch Discovery

This package is the first implementation slice of the MAP2-ROI-driven puncta discovery plan.

Current scope:

- load OME-TIFF image stacks with a lazy-first strategy
- load MAP2 ROI masks generated elsewhere
- sample patch centers relative to the MAP2 ROI
- extract fixed-size local patches and metadata tables
- save disk-backed patch outputs for downstream feature engineering and model training

Current first cohort assumption:

- `ch0`: `MAP2`
- `ch1`: `FLAG`
- `ch2`: `HA`
- `ch3`: `SHANK2`

Biological interpretation for this cohort:

- `MAP2` is positive in all images
- `SHANK2` is positive in all images
- `FLAG` and `HA` are negative in controls and may be present individually or together in experimental conditions

Not yet implemented:

- ND2 to OME-TIFF conversion
- engineered feature extraction
- model training
- embedding export
- novelty scoring

## Install

```powershell
cd C:\Users\Cassandra\Documents\CODE\map2-patch-discovery
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

## Example

```powershell
map2-patch-extract --config configs\example_dataset.yaml
```

Validate a real dataset config without extracting patches:

```powershell
map2-patch-extract --config configs\real_dataset_template.yaml --dry-run
```

Current manifest output includes:

- patch identity and source metadata
- MAP2-relative sampling group
- spatial coordinates and mask-overlap fields
- per-channel summary values: `mean`, `max`, `sum`, `std`, `z_peak_index`, `z_occupancy`
