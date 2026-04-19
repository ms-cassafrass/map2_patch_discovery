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

## External Requirements

For ND2 -> OME-TIFF conversion with Bio-Formats command-line tools:

- `bfconvert.BAT` must be installed
- Java must be installed separately and available on `PATH`

Quick check:

```powershell
java -version
& "C:\path\to\bftools\bfconvert.BAT" -help
```

If `java -version` fails, Bio-Formats conversion will fail even if `bfconvert.BAT`
exists.

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

## MAP2 Mask Export

Use the standalone exporter to generate a usable `*_analysismask.ome.tif`
from the same MAP2-first connectivity-mask design used in the `v5.2` neurite workflow.

Example:

```powershell
map2-mask-export `
  "E:\path\to\image.ome.tif" `
  -o "E:\path\to\mask_output" `
  --channel 0
```

This writes:

- `*_analysismask.ome.tif`
- optional preview PNG

This exporter is self-contained inside `map2-patch-discovery` and does not
import code from the external `Unisynapse` repository at runtime.
